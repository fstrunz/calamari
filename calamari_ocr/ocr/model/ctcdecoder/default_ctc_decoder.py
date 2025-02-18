import numpy as np

from calamari_ocr.ocr.model.ctcdecoder.ctc_decoder import CTCDecoder
from calamari_ocr.ocr.predict.params import Prediction

from typing import Optional


class DefaultCTCDecoder(CTCDecoder):
    def __init__(self, params, codec):
        super().__init__(params, codec)
        self.blank = params.blank_index
        self.threshold = params.min_p_threshold if params.min_p_threshold > 0 else 0.0001

    def decode(self, probabilities) -> Prediction:
        last_char: Optional[int] = None
        chars = np.argmax(probabilities, axis=1)
        sentence = []

        for idx, c in enumerate(chars):
            if c != last_char or c == self.blank:
                # blanks will be contracted into one later on
                sentence.append((c, idx, idx + 1))
            else:
                _, start, _ = sentence[-1]
                del sentence[-1]
                sentence.append((c, start, idx + 1))

            last_char = c

        return self.find_alternatives(probabilities, sentence, self.threshold, self.blank)

    def prob_of_sentence(self, probabilities):
        # do a forward pass and compute the full sentence probability
        pass


if __name__ == "__main__":
    d = DefaultCTCDecoder()
    r = d.decode(
        np.array(
            np.transpose(
                [
                    [0.8, 0, 0.7, 0.2, 0.1],
                    [0.1, 0.4, 0.2, 0.7, 0.8],
                    [0.1, 0.6, 0.1, 0.1, 0.1],
                ]
            )
        )
    )
    print(r)
