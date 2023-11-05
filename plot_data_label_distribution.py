import matplotlib.pyplot as plt

classes = ["Go, Window", "Where, Toilet", "Go, Fridge", "Go, Dustbin", "Open, Oven", "Where, Mirror", "Close, Dishwasher", "Fetch, Computer", "Throw, Cup", "Close, Dustbin", "Open, Fridge", 
 "Go, Couch", "Where, Bed", "Turn, Tap", "Fetch, Pillow", "Where, Cupboard", "Pick, Cloth", "Where, Jug", "Open, Dishwasher", "Close, Microwave", "Go, Sink", "Where, Television", 
 "Where, Bottle", "Throw, Pot", "Fetch, Pot", "Open, Draws", "Go, FirePlace", "Open, Window", "Open, Curtains", "Pick, Bottle", "Turn, Microwave", "Go, Toaster", "Go, Table", 
 "Where, Light", "Fetch, Kettle", "Open, Cupboard", "Fetch, Bowl", "Pick, Jug", "Turn, Oven", "Where, Microwave", "Where, CoffeeMachine", "Go, Desk", 
 "Go, Television", "Where, Sink", "Close, Door", "Pick, Kettle", "Clean, Carpet", "Pick, Computer", "Fetch, Cup", "Throw, Flowers"]

# Split each instance by ", "
parts = [cls.split(", ") for cls in classes]

# Extract the first and second parts
first_parts = [part[0] for part in parts]
second_parts = [part[1] for part in parts]

# Plot the distribution of the first parts
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(first_parts, bins=len(set(first_parts)), edgecolor='k')
plt.xticks(rotation=90)
plt.title('Distribution of First Parts')
plt.xlabel('First Parts')
plt.ylabel('Frequency')

# Plot the distribution of the second parts
plt.subplot(1, 2, 2)
plt.hist(second_parts, bins=len(set(second_parts)), edgecolor='k')
plt.xticks(rotation=90)
plt.title('Distribution of Second Parts')
plt.xlabel('Second Parts')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
