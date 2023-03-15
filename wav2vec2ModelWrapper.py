# Import PyTorchSpeechRecognizer
import logging
from typing import TYPE_CHECKING, List, Optional, Tuple, Union
import numpy as np
from art.estimators.pytorch import PyTorchEstimator
from art.estimators.speech_recognition.speech_recognizer import SpeechRecognizerMixin, PytorchSpeechRecognizerMixin
import torch.nn.functional as F
import torch
# Import wav2vec2.0 model from pytorch models
from pkg_resources import packaging  # type: ignore[attr-defined]
from art import config
from art.utils import get_file

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch

    from art.defences.postprocessor.postprocessor import Postprocessor
    from art.defences.preprocessor.preprocessor import Preprocessor
    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE

logger = logging.getLogger(__name__)

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])

# Define your wrapper class
class wav2vec2Model(PytorchSpeechRecognizerMixin, SpeechRecognizerMixin, PyTorchEstimator):
    # Initialize your wrapper with your model and other parameters
    '''
    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)
    <ipython-input-41-21349aa09a76> in <module>
    ----> 1 estimator = wav2vec2Model( model )

    TypeError: Can't instantiate abstract class wav2vec2Model with abstract methods 
    get_activations, input_shape, loss_gradient 

    '''
    estimator_params = PyTorchEstimator.estimator_params + ["optimizer", "use_amp", "opt_level", "lm_config", "verbose"]
    
    def __init__(
        self,
        model,
        optimizer: Optional["torch.optim.Optimizer"] = None,
        use_amp: bool = False,
        opt_level: str = "O1",
        verbose: bool = True,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union["Preprocessor",List["Preprocessor"],None]=None,
        postprocessing_defences: Union["Postprocessor",List["Postprocessor"],None]=None,
        preprocessing: "PREPROCESSING_TYPE" = None,

    ):
    #bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    #model = bundle.get_model().to(device)   <-- This should be the model input

        super().__init__(
            model = model,
            clip_values=clip_values,
            channels_first=None,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #transcription encoder for CTC LOSS
    def encode_transcription(self, transcription):
        # Define the dictionary
        dictionary = {'-': 0, '|': 1, 'E': 2, 'T': 3, 'A': 4, 'O': 5, 'N': 6, 'I': 7, 'H': 8, 'S': 9, 'R': 10, 'D': 11, 'L': 12, 'U': 13, 'M': 14, 'W': 15, 'C': 16, 'F': 17, 'G': 18, 'Y': 19, 'P': 20, 'B': 21, 'V': 22, 'K': 23, "'": 24, 'X': 25, 'J': 26, 'Q': 27, 'Z': 28}

        # Convert transcription string to list of characters
        chars = list(transcription)

        # Encode each character using the dictionary
        encoded_chars = [dictionary[char] for char in chars]

        # Concatenate the encoded characters to form the final encoded transcription
        encoded_transcription = torch.tensor(encoded_chars)

        return encoded_transcription

    # Implement compute_loss_and_decoded_output method
    def compute_loss_and_decoded_output(
        self, masked_adv_input: "torch.Tensor", original_output: np.ndarray, **kwargs
    ) -> Tuple["torch.Tensor", np.ndarray]:
        """
        Compute loss function and decoded output.
        :param masked_adv_input: The perturbed inputs.
        :param original_output: Target values of shape (nb_samples). Each sample in `original_output` is a string and
                                it may possess different lengths. A possible example of `original_output` could be:
                                `original_output = np.array(['SIXTY ONE'])`
        :param real_lengths: Real lengths of original sequences.
        :return: The loss and the decoded output.
        """

        # Changing the variable name for my convenience 
        x_tensor = masked_adv_input.to(self.device)
        x_tensor = x_tensor.float()
        # Performing inference
        with torch.inference_mode():
            emission, _ = self.model(x_tensor).to(self.device)

        # Decoding the model's output
        decoder = GreedyCTCDecoder(labels = bundle.get_labels())
        transcript = decoder(emission[0])
        transcript = transcript.replace("|"," ")
        
        # Encoding the target transcription
        encoded_transcription = encode_transcription(original_output[0].replace(" ","|"))
        
        # Declaring arguments for CTC Loss
        outputs_ = emission.transpose(0, 1).to(self.device)
        targets = torch.tensor(encoded_transcription, dtype=torch.long).to(self.device)
        output_sizes = torch.tensor([emission.shape[1]], dtype=torch.long).to(self.device)
        target_sizes = torch.tensor([len(encoded_transcription)], dtype=torch.long).to(self.device)

        # Calculating loss
        loss = F.ctc_loss(outputs_, targets, output_sizes, target_sizes) 

        return torch.tensor(loss), transcript.numpy()

   # Implement to_training_mode method 
    def to_training_mode(self) -> None:
      
        # Set your model to training mode 
        self.model.train()

    # Implement sample_rate property 
    @property
    def sample_rate(self) -> int:
      
        # Return the sample rate of your model (you may need to check the documentation or source code of your model)
        return 16000

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.
        :return: Shape of one input sample.
        """
        pass

    def get_activations(
        self, x: np.ndarray, layer: Union[int, str], batch_size: int, framework: bool = False
    ) -> np.ndarray:
        pass

    def loss_gradient(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        pass