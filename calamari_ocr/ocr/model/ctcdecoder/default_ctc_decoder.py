import numpy as np

from calamari_ocr.ocr.model.ctcdecoder.ctc_decoder import CTCDecoder
from calamari_ocr.ocr.predict.params import Prediction

from typing import Optional, Dict


class DefaultCTCDecoder(CTCDecoder):
    def __init__(self, params, codec):
        super().__init__(params, codec)
        self.blank = params.blank_index
        self.threshold = params.min_p_threshold if params.min_p_threshold > 0 else 0.0001

    def decode(self, probabilities) -> Prediction:
        last_char: Optional[int] = None
        chars = np.argmax(probabilities, axis=1)
        sentence = []
        blanks: Dict[int, float] = {}

        for idx, c in enumerate(chars):
            if c != last_char:
                if sentence and c == self.blank:
                    _, start, end = sentence[-1]
                    prob = np.max(probabilities[start:end], axis=0)[self.blank]
                    blanks[start] = prob
                sentence.append((c, idx, idx + 1))
            else:
                # duplicate character, remove it
                _, start, end = sentence[-1]
                prob = np.max(probabilities[start:end], axis=0)[self.blank]

                if last_char == self.blank:
                    if idx - 1 in blanks:
                        prob = max(prob, blanks[idx - 1])
                        del blanks[idx - 1]

                    blanks[idx] = prob

                del sentence[-1]
                sentence.append((c, start, idx + 1))

            last_char = c

        return self.find_alternatives(probabilities, sentence, self.threshold, blanks, self.blank)

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
