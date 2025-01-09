import pickle
import torch.nn as nn
import torch
import transformers
from huggingface_hub import hf_hub_download
import joblib

class VITContrastiveHF(nn.Module):
    def __init__(self, repo_name, classificator_type):
        super(VITContrastiveHF, self).__init__()
        self.model = transformers.AutoModel.from_pretrained(repo_name)
        self.model.pooler = nn.Identity()

        self.processor = transformers.AutoProcessor.from_pretrained(repo_name)
        self.processor.do_resize = False

        # Load the correct classifier
        if classificator_type == 'svm':
            file_path = hf_hub_download(repo_id=repo_name, filename='sklearn/ocsvm_kernel_poly_gamma_auto_nu_0_1_crop.joblib')
            self.classifier = joblib.load(file_path)

        elif classificator_type == 'linear':
            file_path = hf_hub_download(repo_id=repo_name, filename='sklearn/linear_tot_classifier_epoch-32.sav')
            self.classifier = joblib.load(file_path)

        elif classificator_type == 'knn':
            file_path = hf_hub_download(repo_id=repo_name, filename='sklearn/knn_tot_classifier_epoch-32.sav')
            self.classifier = joblib.load(file_path)

        else:
            raise ValueError('Invalid classifier type')

    def forward(self, x, return_feature=False):
        features = self.model(x)
        if return_feature:
            return features
        features = features.last_hidden_state[:, 0, :].cpu().detach().numpy()
        predictions = self.classifier.predict(features)
        return torch.from_numpy(predictions)    

# Example for saving the linear model
model = VITContrastiveHF("aimagelab/CoDE", "linear")
with open("pickles/finalized_model_linear.sav", "wb") as file:
    pickle.dump(model, file)
    
model = VITContrastiveHF("aimagelab/CoDE", "knn")
with open("pickles/finalized_model_knn.sav", "wb") as file:
    pickle.dump(model, file)
    
model = VITContrastiveHF("aimagelab/CoDE", "svm")
with open("pickles/finalized_model_svm.sav", "wb") as file:
    pickle.dump(model, file)
