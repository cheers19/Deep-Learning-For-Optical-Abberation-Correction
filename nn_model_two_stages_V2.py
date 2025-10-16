import torch
import torch.nn as nn

class TwoStageClassifier(nn.Module):
    """
    A two-stage classifier that first predicts error_type,
    then uses the predicted error_type along with the input
    to predict error_element.
    """
    def __init__(self, input_dim, num_error_types, num_error_elements, embed_dim=8):
        super().__init__()

        '''
        # Stage 1: error_type prediction
        self.error_type_classifier = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_error_types)
        )

        '''
        # Stage 1: error_type prediction
        # The input now consists of two parts: x1 (input_dim1) and x2 (input_dim2)
        # input_dim is now combined_input_dim? Let's assume input_dim is now input_dim1
        # Need to update the __init__ to accept input_dim1 and input_dim2
        # Assuming input_dim in the original code corresponds to input_dim1
        input_dim1 = input_dim # Rename for clarity based on the prompt
        input_dim2 = 10 # This needs to be a parameter or infered from somewhere. Assuming a placeholder of 10 for now.
        # Let's update the constructor signature.

    def __init__(self, input_dim1, input_dim2, num_error_types, num_error_elements):
        super().__init__()

        # Layer for input_dim1
        self.fc1_dim1 = nn.Linear(input_dim1, 32)
        self.relu_dim1 = nn.ReLU()
        # Add another 16 length RELU layer for the data of input_dim1
        self.fc2_dim1 = nn.Linear(32, 16)
        self.relu2_dim1 = nn.ReLU()


        # Layer for input_dim2
        self.fc1_dim2 = nn.Linear(input_dim2, 16)
        self.relu_dim2 = nn.ReLU()

        # Add another 16 length RELU layer for the data of input_dim2
        self.fc2_dim2 = nn.Linear(16, 16) # Linear layer before the second ReLU
        self.relu3_dim2 = nn.ReLU() # Second ReLU layer


        # Concatenate the outputs of the two ReLU layers
        # The concatenated output size will be 32 + 16 = 48
        combined_relu_output_dim = 32 + 16 # This should be 16 + 16 = 32 now
        combined_relu_output_dim = 16 + 16 # Corrected concatenated output size

        # Next layer is the linear layer for error_type prediction
        self.error_type_classifier = nn.Linear(combined_relu_output_dim, num_error_types)


        '''
        # Stage 2: error_element prediction conditioned on predicted error_type
        self.type_embedding = nn.Embedding(num_error_types, embed_dim)
        self.error_element_classifier = nn.Sequential(
            nn.Linear(input_dim + embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_error_elements)
        )
        '''
        '''
        # The input size for Stage 2 changes: input_dim + num_error_types
        # (No nn.Embedding is needed)
        # Stage 2: error_element prediction conditioned on Stage 1 logits
        # Input size: original input size + logits size
        combined_input_dim = input_dim + num_error_types
        self.error_element_classifier = nn.Sequential(
            nn.Linear(combined_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_error_elements)
        )
        '''

        # The input size for Stage 2 changes based on the new architecture.
        # The input is the concatenation of input_dim2 and the output after the error_type_classifier
        # The output after error_type_classifier has size num_error_types
        # The size of input_dim2 is input_dim2
        # The concatenated input size for Stage 2 is input_dim2 + num_error_types
        stage2_input_dim = input_dim2 + num_error_types
        self.error_element_classifier = nn.Linear(stage2_input_dim, num_error_elements)


        '''

    def forward(self, x):
        # Stage 1: predict error_type
        error_type_logits = self.error_type_classifier(x)
        error_type_pred = error_type_logits.argmax(dim=1)

        # Stage 2: predict error_element using predicted error_type
        embedded_type = self.type_embedding(error_type_pred)
        x_concat = torch.cat([x, embedded_type], dim=1)
        error_element_logits = self.error_element_classifier(x_concat)

        return error_type_logits, error_element_logits
    '''

    def forward(self, x1, x2): # Accept two inputs
        # Stage 1: predict error_type
        # Pass x1 through its layers
        x1_processed = self.relu_dim1(self.fc1_dim1(x1))
        x1_processed = self.relu2_dim1(self.fc2_dim1(x1_processed))


        # Pass x2 through its layers, including the new ReLU layer
        x2_processed = self.relu_dim2(self.fc1_dim2(x2))
        x2_processed = self.relu3_dim2(self.fc2_dim2(x2_processed))


        # Concatenate the processed inputs
        combined_processed_input = torch.cat([x1_processed, x2_processed], dim=1)

        # Pass the concatenated input through the error_type classifier
        error_type_logits = self.error_type_classifier(combined_processed_input)

        # Stage 2: predict error_element
        # Concatenate input_dim2 (x2) with the error_type_logits
        stage2_input = torch.cat([x2, error_type_logits], dim=1)
        error_element_logits = self.error_element_classifier(stage2_input)

        return error_type_logits, error_element_logits