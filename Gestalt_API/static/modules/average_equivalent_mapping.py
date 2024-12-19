import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import json


class ModifiedNetwork(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(ModifiedNetwork, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.instance_projector = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.feature_dim)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outputs = {}
        z1 = self.instance_projector[0](x)
        outputs['linear1_output'] = z1
        z2 = self.instance_projector[1](z1)
        outputs['relu_output'] = z2
        z3 = self.instance_projector[2](z2)
        outputs['linear2_output'] = z3
        z = nn.functional.normalize(z3, dim=1)
        outputs['normalized_output'] = z
        return z, outputs


class EquivalentWeightsCalculator:
    def __init__(self, model_path, input_dim=20, feature_dim=4):
        # Load the model
        self.model = self.load_model(model_path, input_dim, feature_dim)

    def load_model(self, model_path, input_dim, feature_dim):
        # Initialize the network
        model = ModifiedNetwork(input_dim, feature_dim)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['net'])
        model.eval()
        return model

    def compute_equivalent_weights(self, input_data, W1, b1, W2, b2):
        # z1 = W1 * x + b1
        z1 = np.dot(input_data, W1.T) + b1  # shape: (batch_size, 64)

        # ReLU
        activation = (z1 > 0).astype(float)  # shape: (batch_size, 64)

        #multiply W1 with activation
        W1_masked = W1 * activation[:, :, np.newaxis]  # shape: (batch_size, 64, 20)

        # Multiply W2 with masked W1
        W_eq = np.einsum('jk,bki->bji', W2, W1_masked)  # shape: (batch_size, 4, 20)

        return W_eq  # shape: (batch_size, 4, 20)

    def compute_and_save_equivalent_weights(self, csv_file, output_file_avg='average_equivalent_mapping.json', output_file_all='equivalent_weights_by_tag.json'):
        # Read CSV data
        df = pd.read_csv(csv_file)

        # Extract tag_names from the first column
        tag_names = df.iloc[:, 0].astype(str).values  # Ensure they are strings

        # Extract features (assuming features start from the second column)
        input_data = df.iloc[:, 1:].values.astype(np.float32)  # shape: (num_samples, 20)

        # Get model weights and biases
        linear1 = self.model.instance_projector[0]
        linear2 = self.model.instance_projector[2]

        W1 = linear1.weight.detach().numpy()  # (64, 20)
        b1 = linear1.bias.detach().numpy()    # (64,)
        W2 = linear2.weight.detach().numpy()  # (4, 64)
        b2 = linear2.bias.detach().numpy()    # (4,)

        # Compute equivalent weights for all samples
        W_eq_all = self.compute_batch_equivalent_weights(input_data, W1, b1, W2, b2)

        # Compute average equivalent weights
        W_eq_avg = np.mean(W_eq_all, axis=0)  # (4, 20)

        # Prepare average equivalent weights for JSON
        output_mapping_avg = {
            "output_dimensions": [f"z_{j + 1}" for j in range(W_eq_avg.shape[0])],
            "input_dimensions": df.columns[1:].tolist(),
            "weights": W_eq_avg.tolist()
        }

        # Save average equivalent weights to JSON
        with open(output_file_avg, 'w') as f:
            json.dump(output_mapping_avg, f, indent=4)

        print(f"Average equivalent mapping weights have been saved to '{output_file_avg}'")

        # Prepare equivalent weights for each tag
        # Create a dictionary mapping tag_name to its equivalent weight
        # Each W_eq is of shape (4, 20), convert it to list for JSON serialization
        output_mapping_all = {}
        for tag, W_eq in zip(tag_names, W_eq_all):
            output_mapping_all[tag] = W_eq.tolist()

        # Save all equivalent weights with tag_names to JSON
        with open(output_file_all, 'w') as f:
            json.dump(output_mapping_all, f, indent=4)

        print(f"All equivalent weights have been saved to '{output_file_all}'")

    def compute_batch_equivalent_weights(self, input_data, W1, b1, W2, b2, batch_size=512):
        num_samples = input_data.shape[0]
        num_batches = num_samples // batch_size
        W_eq_list = []

        for i in range(num_batches):
            batch_inputs = input_data[i * batch_size:(i + 1) * batch_size]
            W_eq_batch = self.compute_equivalent_weights(batch_inputs, W1, b1, W2, b2)  # (batch_size, 4, 20)
            W_eq_list.append(W_eq_batch)

        # Handle remaining samples
        if num_samples % batch_size != 0:
            batch_inputs = input_data[num_batches * batch_size:]
            W_eq_batch = self.compute_equivalent_weights(batch_inputs, W1, b1, W2, b2)
            W_eq_list.append(W_eq_batch)

        # Concatenate all batches
        return np.concatenate(W_eq_list, axis=0)  # (num_samples, 4, 20)


