import time

class BehaviorEngine:
    """
    Analyzes track metadata to detect events like Loitering or Zone Entry.
    """
    def __init__(self, zone_config=None):
        self.zone_config = zone_config or {} # { "zone_name": [x1,y1,x2,y2] }
        self.track_states = {} # tid -> { "start_time": t, "zone": "None", "last_seen": t }
        
    def update(self, tracks):
        """
        Process list of tracks and return list of events.
        tracks: list of DeepSort tracks (or similar objects with .to_ltwh(), .track_id)
        """
        current_time = time.time()
        active_tids = set()
        events = []
        
        for track in tracks:
            if not track.is_confirmed(): continue
            tid = track.track_id
            active_tids.add(tid)
            
            bbox = track.to_ltwh()
            x, y, w, h = bbox
            center_x, center_y = x + w/2, y + h/2
            
            # 1. Determine Zone
            current_zone = "General"
            for z_name, z_rect in self.zone_config.items():
                zx1, zy1, zx2, zy2 = z_rect
                if zx1 <= center_x <= zx2 and zy1 <= center_y <= zy2:
                    current_zone = z_name
                    break
            
            # 2. Update State
            if tid not in self.track_states:
                self.track_states[tid] = {
                    "start_time": current_time,
                    "zone": current_zone,
                    "zone_enter_time": current_time,
                    "last_seen": current_time
                }
                # Log usage?
            else:
                state = self.track_states[tid]
                state["last_seen"] = current_time
                
                # Check Zone Transition
                if state["zone"] != current_zone:
                    events.append({
                        "type": "ZONE_CHANGE", 
                        "track_id": tid, 
                        "from": state["zone"], 
                        "to": current_zone,
                        "timestamp": current_time
                    })
                    state["zone"] = current_zone
                    state["zone_enter_time"] = current_time
                
                # Check Loitering
                # If in same zone for > 10 seconds (conceptually)
                # We return a continuous state or a one-time alert?
                # Let's return continuous LOITERING status if > threshold
                duration = current_time - state["zone_enter_time"]
                if duration > 10.0 and current_zone != "General":
                    events.append({
                        "type": "LOITERING", 
                        "track_id": tid, 
                        "zone": current_zone,
                        "duration": duration,
                        "timestamp": current_time
                    })

        # Cleanup
        for tid in list(self.track_states.keys()):
            if tid not in active_tids:
                # Track lost/exited
                last_state = self.track_states[tid]
                events.append({
                    "type": "EXIT",
                    "track_id": tid,
                    "last_zone": last_state["zone"],
                    "duration": last_state["last_seen"] - last_state["start_time"],
                    "timestamp": current_time
                })
                del self.track_states[tid]
                
        return events
