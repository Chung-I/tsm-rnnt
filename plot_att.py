import matplotlib.pyplot as plt
import json
import numpy as np

with open("valid_out.txt") as fp:
    outputs = json.load(fp)
if "ctc_predicted_tokens" in outputs:
    print("ctc: " + str(outputs["ctc_predicted_tokens"]))
if "rnnt_predicted_tokens" in outputs:
    print("rnnt: " + str(outputs["rnnt_predicted_tokens"]))
if "predicted_tokens" in outputs:
    print(outputs["predicted_tokens"])
if "attentions" in outputs:
    matrix = np.asarray(outputs["attentions"], dtype=np.float32)
    fig, ax = plt.subplots()
    im = ax.matshow(matrix)
    
    fig.tight_layout()
    plt.savefig("att.png")
