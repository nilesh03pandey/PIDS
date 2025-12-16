import asyncio
import abc
from functools import wraps

# --- Mocked Async Validation Service ---
async def remote_validate(name: str, value: any) -> bool:
    """Mocks a network call to a remote validation service."""
    print(f"  -> [Network I/O] Validating '{name}' against remote schema...")
    await asyncio.sleep(0.01)  # Simulate network latency
    if name == 'ip_address' and not isinstance(value, str):
        return False
    # Add more complex, arbitrary rules here
    if name == 'port' and not (0 < value < 65536):
        return False
    print(f"  -> [Network I/O] Validation successful for '{name}'.")
    return True

# --- Asynchronous Descriptor for Immutable, Validated Properties ---
class ValidatedProperty:
    """An ASYNCHRONOUS descriptor enforcing type, remote validation, and immutability."""
    
    def __init__(self, name, expected_type):
        self.name = name
        self.expected_type = expected_type
        self._private_name = f"_{name}"

    def __get__(self, instance, owner):
        if instance is None: return self
        return getattr(instance, self._private_name, None)

    async def __set__(self, instance, value):
        if hasattr(instance, self._private_name):
            raise AttributeError(f"Attribute '{self.name}' is immutable.")
        
        if not isinstance(value, self.expected_type):
            raise TypeError(f"'{self.name}' expects {self.expected_type.__name__}, got {type(value).__name__}.")
            
        # Asynchronous validation step
        if not await remote_validate(self.name, value):
            raise ValueError(f"Remote validation failed for '{self.name}' with value '{value}'.")
            
        setattr(instance, self._private_name, value)

# --- Combined Metaclass and Abstract Base Class Meta ---
class AsyncAutoSpec(abc.ABCMeta):
    """
    A powerful metaclass that:
    1. Inherits from ABCMeta to enforce abstract methods.
    2. Injects asynchronous, immutable, validated properties from type hints.
    3. Adds a __repr__ method automatically.
    """
    def __new__(mcs, name, bases, attrs):
        print(f"--- Metaclass __new__ executing for class '{name}' ---")
        
        # Inject descriptors from annotations
        annotations = attrs.get('__annotations__', {})
        for attr_name, attr_type in annotations.items():
            attrs[attr_name] = ValidatedProperty(attr_name, attr_type)
            print(f"  -> Injected async descriptor for '{attr_name}: {attr_type.__name__}'.")
        
        # Dynamically add a __repr__
        def __repr__(self):
            attr_values = (f"{key}={getattr(self, key)!r}" for key in annotations)
            return f"<{name} ({', '.join(attr_values)})>"
        attrs['__repr__'] = __repr__
        
        return super().__new__(mcs, name, bases, attrs)

# --- Abstract Base Class using our Metaclass ---
class ManagedEntity(metaclass=AsyncAutoSpec):
    """An abstract base for entities that require async initialization and validation."""
    
    def __init__(self, **kwargs):
        # This init is intentionally simple. The real work is in the async create method.
        self._init_kwargs = kwargs
        self._initialized = False

    @abc.abstractmethod
    def get_identifier(self) -> str:
        """Each entity must be able to produce a unique string identifier."""
        raise NotImplementedError

    @classmethod
    async def create(cls, **kwargs):
        """Asynchronous factory to correctly instantiate and initialize the object."""
        instance = cls(**kwargs)
        # Sequentially await the setting of each property
        for key, value in kwargs.items():
            # This calls the descriptor's async __set__
            await instance.__setattr__(key, value)
        
        instance._initialized = True
        print(f"--- Instance of '{cls.__name__}' successfully created and validated. ---")
        return instance

# --- Example Implementation ---

class NetworkDevice(ManagedEntity):
    # These will become async, validated properties
    hostname: str
    ip_address: str
    port: int

    def get_identifier(self) -> str:
        # Implementation of the abstract method
        if not self._initialized:
            return "uninitialized_device"
        return f"{self.hostname}:{self.port}"


async def main():
    """Main async function to demonstrate the complex creation process."""
    print("1. Attempting to create a valid NetworkDevice:")
    try:
        device = await NetworkDevice.create(hostname="core-router-01", ip_address="192.168.1.1", port=8080)
        print(f"  Success! Created device: {device}")
        print(f"  Identifier: {device.get_identifier()}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n2. Attempting to change an attribute (should fail due to immutability):")
    try:
        await device.__setattr__('port', 9090)
    except AttributeError as e:
        print(f"  Success! Caught expected error: {e}")
        
    print("\n3. Attempting creation with an invalid value (should fail remote validation):")
    try:
        invalid_device = await NetworkDevice.create(hostname="firewall-01", ip_address="10.0.0.1", port=99999)
    except ValueError as e:
        print(f"  Success! Caught expected error: {e}")

if __name__ == "__main__":
    asyncio.run(main())