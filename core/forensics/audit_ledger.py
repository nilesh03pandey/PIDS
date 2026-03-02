import json
import hashlib
import time
import os
from datetime import datetime

class AuditLedger:
    """
    Tamper-Evident Audit Log using Hash Chaining.
    Each entry includes the hash of the previous entry, forming a blockchain-like structure.
    """
    def __init__(self, log_file="audit_ledger.jsonl"):
        self.log_file = log_file
        self.last_hash = self._get_last_hash()
        
    def _get_last_hash(self):
        """Read the last line of the log file to get the previous hash."""
        if not os.path.exists(self.log_file):
            return "0" * 64 # Genesis hash
            
        try:
            with open(self.log_file, 'rb') as f:
                try:
                    # Seek to end and read backwards to find last newline?
                    # Simplified: just read lines. For large files, seek is better.
                    # But for now, simple readlines is safer against corruption.
                    lines = f.readlines()
                    if not lines: return "0" * 64
                    last_line = lines[-1].decode('utf-8').strip()
                    if not last_line: return "0" * 64
                    entry = json.loads(last_line)
                    return entry.get("current_hash", "0"*64)
                except Exception:
                    return "0" * 64
        except Exception:
            return "0" * 64

    def log(self, event_type, details, actor="system"):
        """
        Log an event with cryptographic integrity.
        """
        timestamp = datetime.utcnow().isoformat()
        
        # Construct the payload
        payload = {
            "timestamp": timestamp,
            "event_type": event_type,
            "actor": actor,
            "details": details,
            "prev_hash": self.last_hash
        }
        
        # Calculate Hash
        # Canonical JSON string representation for hashing ensures consistency
        payload_str = json.dumps(payload, sort_keys=True)
        current_hash = hashlib.sha256(payload_str.encode('utf-8')).hexdigest()
        
        payload["current_hash"] = current_hash
        
        # Write to file
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(payload) + "\n")
            
        self.last_hash = current_hash
        return current_hash
        
    def verify_integrity(self):
        """
        Verify the entire chain. Returns (True, None) or (False, ErrorMsg).
        """
        if not os.path.exists(self.log_file):
            return True, "No log file."
            
        prev_h = "0" * 64
        line_num = 0
        
        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                line_num += 1
                line = line.strip()
                if not line: continue
                
                try:
                    entry = json.loads(line)
                    stored_hash = entry.get("current_hash")
                    stored_prev_hash = entry.get("prev_hash")
                    
                    # 1. Check Link
                    if stored_prev_hash != prev_h:
                        return False, f"Broken chain at line {line_num}: prev_hash mismatch."
                        
                    # 2. Check Content Integrity
                    # Reconstruct payload for hashing (remove current_hash)
                    verify_payload = entry.copy()
                    del verify_payload["current_hash"]
                    
                    recalc_str = json.dumps(verify_payload, sort_keys=True)
                    recalc_hash = hashlib.sha256(recalc_str.encode('utf-8')).hexdigest()
                    
                    if recalc_hash != stored_hash:
                        return False, f"Content tampering at line {line_num}."
                        
                    prev_h = stored_hash
                    
                except json.JSONDecodeError:
                    return False, f"Corrupt JSON at line {line_num}."
                    
        return True, "Integrity Verified."
