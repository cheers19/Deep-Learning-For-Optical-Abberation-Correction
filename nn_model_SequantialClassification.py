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

        # Stage 1: error_type prediction
        self.error_type_classifier = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_error_types)
        )


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


    def forward(self, x):
        # Stage 1: predict error_type
        error_type_logits = self.error_type_classifier(x)

        # Note: argmax is not required here, unless you need it for calculating the accuracy of Stage 1.
        
        # If needed:
        # error_type_pred = error_type_logits.argmax(dim=1)

        # Stage 2: predict error_element by concatenating the original input with the logits
        # Direct use of error_type_logits as the conditional representation
        x_concat = torch.cat([x, error_type_logits], dim=1)
        error_element_logits = self.error_element_classifier(x_concat)

        return error_type_logits, error_element_logits

