import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.utils import save_image
import numpy as np

# Define the Generator network
class Generator(nn.Module):
    def __init__(self, input_size=100, hidden_size=256, output_size=784):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Define the Discriminator network
class Discriminator(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, output_size=1):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Define the hyperparameters
input_size = 100
hidden_size = 256
output_size = 784
num_epochs = 200
batch_size = 128
learning_rate = 0.0002
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the MNIST dataset
transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
mnist = MNIST(root="data", download=True, transform=transform)
dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True)

# Create the Generator and Discriminator networks
generator = Generator(input_size, hidden_size, output_size).to(device)
discriminator = Discriminator(input_size, hidden_size, output_size).to(device)

# Define the loss function and optimizers
criterion = nn.BCELoss()
gen_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
dis_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)

# Define a function to generate random noise vectors
def generate_noise(batch_size, input_size):
    return torch.randn(batch_size, input_size).to(device)

# Train the GAN
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(dataloader):
        batch_size = images.size(0)
        images = images.view(batch_size, -1).to(device)
        
        # Train the Discriminator network
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Train the Discriminator with real images
        outputs = discriminator(images)
        dis_loss_real = criterion(outputs, real_labels)
        dis_optimizer.zero_grad()
        dis_loss_real.backward()

        # Train the Discriminator with fake images
        noise = generate_noise(batch_size, input_size)
        fake_images = generator(noise)
        outputs = discriminator(fake_images)
        dis_loss_fake = criterion(outputs, fake_labels)
        dis_optimizer.zero_grad()
        dis_loss_fake.backward()

        # Update the Discriminator weights
        dis_optimizer.step()

        # Train the Generator network
        noise = generate_noise(batch_size, input_size)
        fake_images = generator(noise)
        outputs = discriminator(fake_images)
        gen_loss = criterion(outputs, real_labels)

        # Backpropagate and update the Generator weights
        gen_optimizer.zero_grad()
        gen_loss.backward()
        gen_optimizer.step()

        # Print the loss and save the generated images every 100 batches
        if (i+1) % 100 == 0:
            print("Epoch [{}/{}], Batch [{}/{}], Discriminator Loss: {:.4f}, Generator Loss: {:.4f}"
                  .format(epoch+1, num_epochs, i+1, len(dataloader), (dis_loss_real+dis_loss_fake)/2, gen_loss))
            
            # Save the generated images
            with torch.no_grad():
                fake_images = generator(generate_noise(64, input_size)).cpu()
                save_image(fake_images.view(64, 1, 28, 28), "output/{}_{}.png".format(epoch+1, i+1))

# Save the models
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")
