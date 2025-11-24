from braket.circuits import Circuit
from braket.aws import AwsDevice, AwsSession
from braket.devices import LocalSimulator
import boto3

# Create a tripartite GHZ circuit
# GHZ state: (|000⟩ + |111⟩) / √2
circuit = Circuit()

# Apply Hadamard gate to first qubit
circuit.h(0)

# Apply CNOT gates to create entanglement
circuit.cnot(0, 1)
circuit.cnot(1, 2)

# Add measurement operations
circuit.measure([0, 1, 2])

print("Circuit diagram:")
print(circuit)

# 1: Run on local simulator
# device = LocalSimulator()
# task = device.run(circuit, shots=1000)
# result = task.result()
#
# print("\nMeasurement counts:")
# print(result.measurement_counts)

# 2: Run on AWS Braket quantum device
# session = AwsSession(boto_session=boto3.Session(region_name='eu-north-1'))
session = AwsSession(boto_session=boto3.Session(region_name='us-east-1'))
device = AwsDevice(
    "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
    aws_session=session
)
task = device.run(circuit, shots=100)
result = task.result()
print("\nMeasurement counts:")
print(result.measurement_counts)
