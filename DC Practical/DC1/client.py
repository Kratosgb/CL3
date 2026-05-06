import xmlrpc.client

# Connect to server
client = xmlrpc.client.ServerProxy("http://localhost:8000/")

# Input from user
num = int(input("Enter a number: "))

# Remote call
result = client.factorial(num)

# Output
print("Factorial:", result)