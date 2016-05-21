import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import pickle, gzip
from collections import defaultdict
import utilsFunctions
import createFingerprintsReaction
from flask import Flask, render_template, request
from wtforms import Form, TextField, validators


with gzip.open('result_lr_fp_AP3_MG2.pkl.gz', 'rb') as infile:
    result_lr_fp_AP3_MG2,rtypes,names_rTypes = pickle.load(infile)

def predict(smi):
    fpsz = 512  # TODO: connect this value with ipynb fp generator
    rxn = AllChem.ReactionFromSmarts(smi,useSmiles=True)
    fp_AP3 = createFingerprintsReaction.create_transformation_FP(rxn,AllChem.FingerprintType.AtomPairFP)
    fp_MG2_agents = createFingerprintsReaction.create_agent_morgan2_FP(rxn)
    if fp_MG2_agents is None:
        fp_MG2_agents = DataStructs.UIntSparseIntVect(4096)
    np1_morgan = utilsFunctions.fpToNP(fp_AP3,fpsz)
    np2_morgan = utilsFunctions.fpToNP(fp_MG2_agents,fpsz)
    testFps_AP3_agentMG2 = [np.concatenate([np1_morgan, np2_morgan])]
    predict = result_lr_fp_AP3_MG2.predict(testFps_AP3_agentMG2)
    answer = rtypes[predict[0]] + ' ' + names_rTypes[rtypes[predict[0]]]
    return answer


app = Flask(__name__)

class InputForm(Form):
    smi = TextField(validators=[validators.InputRequired()])

@app.route('/', methods=['GET', 'POST'])
def test():
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
        smi = form.smi.data
        result = predict(smi)
    else:
        result = None

    return render_template("test.html", form=form, result=result)

if __name__ == '__main__':
    app.run(debug=True)