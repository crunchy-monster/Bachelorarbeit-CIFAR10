#================================================================0
#==============================
# Before training hyperparameters neet to be set

model = LeNet5(num_classes).to(device)

# Setting the loss function
cost = nn.CrossEntropyLoss()

# Setting the optimizer with the model parameters and learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# this is defined to print how many steps are remaining when training
total_step = len(train_loader)

total_step = len(train_loader)
for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):
		images = images.to(device)
		labels = labels.to(device)

		# Forward pass
		outputs = model(images)
		loss = cost(outputs, labels)
		# Backward and optimize
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if (i + 1) % 400 == 0:
			print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step,
			                                                         loss.item()))

	# Test the model
# In the test phase, we don't need to compute gradients (for memory efficiency)

model.eval()  # Set the model to evaluation mode

with torch.no_grad():
	correct = 0
	total = 0

	for images, labels in test_loader:
		images = images.to(device)
		labels = labels.to(device)

		outputs = model(images)
		_, predicted = torch.max(outputs.data, 1)

		total += labels.size(0)
		correct += (predicted == labels).sum().item()

	accuracy = 100 * correct / total
	print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f} %')

