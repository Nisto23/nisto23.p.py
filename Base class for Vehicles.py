# Base class for Vehicles
class Vehicle:
    def __init__(self, name, speed):
        self.name = name
        self.speed = speed
    
    def move(self):
        """A generic move method to be overridden."""
        print(f"{self.name} is moving at {self.speed} km/h.")

# Subclass for Car
class Car(Vehicle):
    def __init__(self, name, speed, fuel_type):
        super().__init__(name, speed)
        self.fuel_type = fuel_type
    
    def move(self):
        print(f"{self.name} is driving on the road at {self.speed} km/h, using {self.fuel_type} fuel.")

# Subclass for Plane
class Plane(Vehicle):
    def __init__(self, name, speed, altitude):
        super().__init__(name, speed)
        self.altitude = altitude
    
    def move(self):
        print(f"{self.name} is flying in the air at {self.speed} km/h at an altitude of {self.altitude} meters.")

# Subclass for Boat
class Boat(Vehicle):
    def __init__(self, name, speed, capacity):
        super().__init__(name, speed)
        self.capacity = capacity
    
    def move(self):
        print(f"{self.name} is sailing on the water at {self.speed} km/h, with a capacity of {self.capacity} passengers.")

# Example of polymorphism
def demonstrate_polymorphism(vehicle):
    vehicle.move()

# Create instances of each vehicle type
car = Car(name="Sedan", speed=120, fuel_type="Gasoline")
plane = Plane(name="Jet", speed=800, altitude=10000)
boat = Boat(name="Yacht", speed=50, capacity=20)

# Demonstrate polymorphism
vehicles = [car, plane, boat]
for v in vehicles:
    demonstrate_polymorphism(v)
