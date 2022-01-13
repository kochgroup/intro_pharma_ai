from IPython.display import Image, HTML
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import Draw, AllChem
from rdkit import Chem
from tqdm import tqdm
from torch import nn
import torch
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt

#%% Plot SVC

def imshow(inp, title=None):
    """Imshow for Tensor.
    Credit: PyTorch Tutorial: Transfer Learning
    """
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def plot_svc(data, model):
    """
    Credit:  Newbedev  - https://github.com/newbedev-com
    """
    
    fig, ax = plt.subplots()
    # title for the plots
    title = ('Decision surface of linear SVC ')
    # Set-up grid for plotting.
    X0 = data.x1
    X1 = data.x2
    xx, yy = make_meshgrid(X0, X1)

    plot_contours(ax, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=data.y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_ylabel('x2')
    ax.set_xlabel('x1')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    plt.show()

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - ((np.abs(x.min())+0.1)*0.1), x.max() + (np.abs(x.max())*0.1) 
    y_min, y_max = y.min() - ((np.abs(y.min())+0.1)*0.1), y.max() + (np.abs(y.max())*0.1)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


#%%

def get_fingerprints(data, bitSize_circular=2048, labels_default=None , labels_morgan=None, morgan_radius=2):
    
    """ Computes the Fingerprints from Molecules
    """
    feature_matrix= pd.DataFrame(np.zeros((data.shape[0],bitSize_circular)), dtype=int) 
    for i in tqdm(range(data.shape[0])):
       feature_matrix.iloc[i,:] = np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data.smiles.iloc[i]),morgan_radius,nBits=bitSize_circular)) 


    return(feature_matrix)



def showBit(mol, nBit, bi):
    atomId, radius = bi[nBit][0]
    molSize = (150,150)
    drawOptions=None
    menv=Draw._getMorganEnv(mol,atomId,radius, molSize=molSize, baseRad=0.3, aromaticColor=(0.9, 0.9, 0.2), ringColor=(0.8, 0.8, 0.8),
                      centerColor=(0.6, 0.6, 0.9), extraColor=(0.9, 0.9, 0.9))
    
    
    
    drawer = rdMolDraw2D.MolDraw2DCairo(molSize[0], molSize[1])
    if drawOptions is None:
      drawOptions = drawer.drawOptions()
    drawOptions.continuousHighlight = False
    drawOptions.includeMetadata = False
    drawer.SetDrawOptions(drawOptions)
    drawer.DrawMolecule(menv.submol, highlightAtoms=menv.highlightAtoms,
                        highlightAtomColors=menv.atomColors, highlightBonds=menv.highlightBonds,
                        highlightBondColors=menv.bondColors, highlightAtomRadii=menv.highlightRadii)
    drawer.FinishDrawing()
    return Image(drawer.GetDrawingText())


def hide_toggle(for_next=False):
    this_cell = """$('div.cell.code_cell.rendered.selected')"""
    next_cell = this_cell + '.next()'

    toggle_text = 'Toggle show/hide'  # text shown on toggle link
    target_cell = this_cell  # target cell to control with toggle
    js_hide_current = ''  # bit of JS to permanently hide code in current cell (only when toggling next cell)

    if for_next:
        target_cell = next_cell
        toggle_text += ' next cell'
        js_hide_current = this_cell + '.find("div.input").hide();'

    js_f_name = 'code_toggle_{}'.format(str(random.randint(1,2**64)))

    html = """
        <script>
            function {f_name}() {{
                {cell_selector}.find('div.input').toggle();
            }}

            {js_hide_current}
        </script>

        <a href="javascript:{f_name}()">{toggle_text}</a>
    """.format(
        f_name=js_f_name,
        cell_selector=target_cell,
        js_hide_current=js_hide_current, 
        toggle_text=toggle_text
    )

    return HTML(html)

def create_dict(smiles, add_tokens=False):
    vocabulary = []    
    if add_tokens: 
        vocabulary=["<sos>", "<eos>", "<pad>"]
    for smile in smiles:
        atoms = []
        i=0
        while i < len(smile):
            if smile[i:i+2]== "Br":
                atoms.append("Br")
                i+=2
            elif smile[i:i+2]== "Cl":
                atoms.append("Cl")
                i+=2
            elif smile[i:i+2]== "Si":
                atoms.append("Si")
                i+=2
            elif smile[i:i+2]== "Cr":
                atoms.append("Cr")
                i+=2   
            else:
                atoms.append(smile[i])
                i+=1
                
        vocabulary += list(set(atoms)-set(vocabulary))
    return {vocabulary[i]: i for i in range(len(vocabulary))}   


def tokenize(smiles, dictionary, add_tokens = False):
    token_smiles = []
    for smile in smiles:
        token_smile= []
        i = 0
        
        if add_tokens:
            token_smile.append(dictionary["<sos>"])    
        while i < len(smile):
            if (smile[i:i+2]=="Cl"):
                token_smile.append(dictionary["Cl"])
                i+=2
            elif (smile[i:i+2]=="Br"):
                token_smile.append(dictionary["Br"])
                i+=2
            elif (smile[i:i+2]=="Si"):
                token_smile.append(dictionary["Si"])
                i+=2
            elif (smile[i:i+2]=="Cr"):
                token_smile.append(dictionary["Cr"])
                i+=2
            else:
                token_smile.append(dictionary[smile[i]])
                i+=1
        if add_tokens:
            token_smile.append(dictionary["<eos>"])
        token_smiles.append(token_smile)   
    return token_smiles


def token_to_onehot(tokenized_smiles, vocabulary_length):
    one_hot_ll = list()
    for smile in tokenized_smiles:
        one_hot_matrix=np.zeros([len(smile),vocabulary_length])
        for i, token in enumerate(smile):
            one_hot_matrix[i,token]=1
        one_hot_ll.append(one_hot_matrix)
    return np.stack(one_hot_ll)


class Permute(nn.Module):
    def __init__(self,*args):
        super(Permute, self).__init__()
        self.shape = args
        
    def forward(self, x):
        return x.permute(self.shape)


def tokens_to_smiles(tokens, dictionary):
    inv_dictionary = {dictionary[k] : k for k in dictionary}
    smiles_ll = []
    for i,mol in enumerate(tokens):
        smile_ll = ""
        for k in range(1, mol.shape[0]):
            if inv_dictionary[mol[k]] == "<eos>":
                break
            else:
                smile_ll+=inv_dictionary[mol[k]]

        smiles_ll.append(smile_ll)
    return smiles_ll

def evaluate(model, loader, dictionary):
    model.eval()
    perc_valid_ll  = []
    perc_ident_ll = []
    for input_seq, output_seq in iter(loader):
        input_seq = input_seq.t()
        output_seq = output_seq.t()
        pred = model(input_seq, output_seq, 0.)
        pred_tokens = torch.argmax(pred,2).t().detach().numpy()

        smiles_pred = tokens_to_smiles(pred_tokens, dictionary)
        smiles_true = tokens_to_smiles(output_seq.t().detach().numpy(), dictionary)

        valid_smiles_count = 0
        for smile in smiles_pred:
            if Chem.MolFromSmiles(smile) != None:
                valid_smiles_count +=1

        valid_smiles_count /= len(smiles_true)
        perc_valid_ll.append(valid_smiles_count)

        smiles_ident = 0
        for i in range(len(smiles_pred)):
            if smiles_pred[i] == smiles_true[i]:
                smiles_ident += 1

        perc_ident_ll.append(smiles_ident/len(smiles_pred))   

    return np.sum(perc_valid_ll)/len(loader), np.sum(perc_ident_ll)/len(loader)

