from IPython.display import Image, HTML
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import Draw, AllChem
from rdkit import Chem
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt

#%% Plot SVC

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