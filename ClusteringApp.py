import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import py3Dmol
import tempfile
import tkinter as tk
import ttkbootstrap as tb
import webbrowser
from kneed import KneeLocator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from openeye import oechem
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler
from tkinter import filedialog, messagebox
from ttkbootstrap.constants import *


def checkContent(filePath):
    # Create an input stream for reading a molecular data file
    fileStream = oechem.oemolistream()
    # Specify the OEB format for the input file
    fileStream.SetFormat(oechem.OEFormat_OEB)
    # Enable automatic decompression if the input file is gzipped
    fileStream.Setgz(True)
    fileStream.open(filePath)

    # Check if the input file contains at least one molecule
    for mol in fileStream.GetOEMols():
        fileStream.close()
        return True
    
    fileStream.close()
    return False


def checkKeywords(df):
    # Check if the input file (as a dataframe) contains all required fields
    keywords = ["title", "donor", "acceptor", "rotatable", "logp", "tpsa", "mw", "smiles", "fred chemgauss4 score"]
    lowerCaseCols = [col.lower() for col in df.columns]
    return all(any(keyword in col for col in lowerCaseCols) for keyword in keywords)


def constructDataFrame(filePath):
    fileStream = oechem.oemolistream()
    fileStream.SetFormat(oechem.OEFormat_OEB)
    fileStream.Setgz(True)
    fileStream.open(filePath)

    dataList = []

    for mol in fileStream.GetOEMols():
        data = {"Title": mol.GetTitle()}

        # Process Structure-Data tags
        for each in oechem.OEGetSDDataPairs(mol):
            tag, value = each.GetTag().strip(), each.GetValue().strip()

            if tag.lower() == "mw" or tag.lower() == "smiles":
                continue

            data[tag] = value

        data["MW"] = oechem.OECalculateMolecularWeight(mol)
        data["SMILES"] = oechem.OEMolToSmiles(mol)

        # Extract the Chemgauss4 score
        for conf in mol.GetConfs():
            if oechem.OEHasSDData(conf, "FRED Chemgauss4 score"):
                data["FRED Chemgauss4 score"] = oechem.OEGetSDData(conf, "FRED Chemgauss4 score")
                break

        dataList.append(data)

    fileStream.close()
    return pd.DataFrame(dataList)


def fingerprintToNumpy(fp):
    # Create an array of zeros with a length equal to the number of bits in the fingerprint
    array = np.zeros((fp.GetNumBits(),), dtype = int)
    DataStructs.ConvertToNumpyArray(fp, array)
    return array


def processAndClusterData(df):
    # ------------------
    # Data preprocessing
    # ------------------
    keywords = ["title", "donor", "acceptor", "rotatable", "tpsa", "mw", "smiles", "fred chemgauss4 score"]

    # Determine the type of logP
    logpKeyword = None
    logpType = None

    for col in df.columns:
        if "clogp" in col.lower():
            logpKeyword = "clogp"
            logpType = "clogP"
            break

        if "logp" in col.lower():
            logpKeyword = "logp"
            logpType = "logP"
            break

    keywords.insert(keywords.index("tpsa"), logpKeyword)

    # Filter columns
    filteredCols = []

    for k in keywords:
        for col in df.columns:                
            if k in col.lower():
                filteredCols.append(col)      
                break

    filteredDf = df[filteredCols].copy()

    # Rename columns for consistency
    renamedCols = ["title", "donors", "acceptors", "rotatable", logpType, "TPSA", "MW", "SMILES", "FRED Chemgauss4 score"]
    renameMap = dict(zip(filteredCols, renamedCols))
    filteredDf.rename(columns = renameMap, inplace = True)

    # Drop rows with missing values
    filteredDf = filteredDf.dropna().reset_index(drop = True)

    # Ensure data types are correct
    filteredDf["donors"] = filteredDf["donors"].astype(int)
    filteredDf["acceptors"] = filteredDf["acceptors"].astype(int)
    filteredDf["rotatable"] = filteredDf["rotatable"].astype(int)
    filteredDf[logpType] = filteredDf[logpType].astype(float)
    filteredDf["TPSA"] = filteredDf["TPSA"].astype(float)
    filteredDf["MW"] = filteredDf["MW"].astype(float)
    filteredDf["FRED Chemgauss4 score"] = filteredDf["FRED Chemgauss4 score"].astype(float)

    # Convert SMILES strings to RDKit Mol objects
    filteredDf["RDKit Mol"] = filteredDf["SMILES"].apply(lambda smiles: Chem.MolFromSmiles(smiles))
    # Generate RDKit fingerprints from molecule objects
    filteredDf["fingerprint"] = filteredDf["RDKit Mol"].apply(lambda mol: Chem.RDKFingerprint(mol))
    # Convert each fingerprint to a 1D NumPy bit vector
    filteredDf["fingerprint"] = filteredDf["fingerprint"].apply(lambda fp: fingerprintToNumpy(fp))

    filteredDf = filteredDf.drop(columns = ["RDKit Mol"])

    # Scale numerical features
    scaler = RobustScaler()
    numericalFeatures = ["donors", "acceptors", "rotatable", logpType, "TPSA", "MW"]
    scaledFeatures = scaler.fit_transform(filteredDf[numericalFeatures])

    # Stack fingerprint vectors into a 2D matrix
    fingerprintMatrix = np.vstack(filteredDf["fingerprint"].values)

    X = np.hstack([scaledFeatures, fingerprintMatrix])

    # Determine the optimal number of components for PCA
    eigenvalues = PCA().fit(X).explained_variance_
    kneedle = KneeLocator(range(1, 25 + 1), eigenvalues[:25], curve = "convex", direction = "decreasing")
    elbow = kneedle.elbow

    # Perform PCA to reduce dimensionality
    pca = PCA(n_components = elbow, random_state = 0)
    xPca = pca.fit_transform(X)

    # ----------------------
    # Clustering with DBSCAN
    # ----------------------
    # Tune DBSCAN hyperparameters
    epsRange = np.arange(1, 8, 0.05)
    minSamplesRange = range(2 * xPca.shape[1], 2 * xPca.shape[1] + 5 + 1)
    noiseCap = 0.5
    results = []

    for eps in epsRange:
        for minSamples in minSamplesRange:
            dbscan = DBSCAN(eps = eps, min_samples = minSamples)
            dbscan.fit(xPca)
            labels = dbscan.labels_
            
            numClusters = len(set(labels)) - (1 if -1 in labels else 0)
            numNoise = list(labels).count(-1)

            if numNoise / len(labels) <= noiseCap:
                if numClusters > 1:
                    silhouette = silhouette_score(xPca[labels != -1], labels[labels != -1])
                    results.append((eps, minSamples, numClusters, numNoise, silhouette, labels))

    if results:
        # Get the best result where the silhouette score is the highest and if there is a tie, pick the one with the fewest noise points
        bestResult = max(results, key = lambda r: (r[4], -r[3]))
        bestEps, bestMinSamples, bestNumClusters, bestNumNoise, bestSilhouette, bestLabels = bestResult
    else:
        return
    
    # Sort molecules for each cluster, excluding noise points (first in the cluster being the representative)
    pcaCols = ["PC" + str(i + 1) for i in range(xPca.shape[1])]
    pcaDf = pd.DataFrame(xPca, columns = pcaCols)
    pcaDf[numericalFeatures] = scaledFeatures
    pcaDf["SMILES"] = filteredDf["SMILES"].values
    pcaDf["title"] = filteredDf["title"].values
    pcaDf["FRED Chemgauss4 score"] = filteredDf["FRED Chemgauss4 score"].values
    pcaDf["cluster label"] = bestLabels

    rows = []

    for label in set(pcaDf["cluster label"]) - {-1}:
        subset = pcaDf[pcaDf["cluster label"] == label].copy()
        centre = subset[pcaCols].mean().values
        # Calculate distances from the centre
        subset["distance"] = np.linalg.norm(subset[pcaCols] - centre, axis = 1)
        subset = subset.sort_values("distance").reset_index(drop = True)

        subset[numericalFeatures] = scaler.inverse_transform(subset[numericalFeatures])
        rows.append(subset[["title", *numericalFeatures, "SMILES", "FRED Chemgauss4 score", "cluster label"]])

    sortedDf = pd.concat(rows, ignore_index = True)
    return xPca, bestLabels, sortedDf


class ClusteringApp(tb.Window):
    def __init__(self):
        super().__init__(themename = "superhero")
        self.title("Docked Compound Cluster Visualiser")

        self.filePath = tk.StringVar()
        self.resultsDf = pd.DataFrame()

        self.fig = None
        self.ax = None
        self.canvas = None
        self.plotFrame = None

        # Maximize the window to fill the entire screen
        self.state("zoomed")

        self._buildUI()


    def _buildUI(self):
        self.topSectionFrame = tb.Frame(self)
        self.topSectionFrame.pack(pady = (100, 0))

        # Add a title
        tb.Label(self.topSectionFrame, text = "Docked Compound Cluster Visualiser", font = ("Helvetica", 20)).pack(pady = (10, 5))

        inputFrame = tb.Frame(self.topSectionFrame)
        inputFrame.pack(pady = 5)

        # Add a label, read-only entry and browse button
        tb.Label(inputFrame, text = "Select OEB File:").pack(side = "left", padx = 5)
        tb.Entry(inputFrame, textvariable = self.filePath, width = 50, state = "readonly").pack(side = "left", padx = 5)
        tb.Button(inputFrame, text = "Browse", bootstyle = "info", command = self._browse).pack(side = "left", padx = 5)

        buttonFrame = tb.Frame(self.topSectionFrame)
        buttonFrame.pack(pady = 5)

        # Add key buttons
        self.runButt = tb.Button(buttonFrame, text = "Run Clustering", bootstyle = "success", command = self._run, state = DISABLED)
        self.runButt.pack(side = "left", padx = 5)

        self.saveButt = tb.Button(buttonFrame, text = "Save Results", bootstyle = "success", command = self._save, state = DISABLED)
        self.saveButt.pack(side = "left", padx = 5)

        self.view3DButt = tb.Button(buttonFrame, text = "View Molecules in 3D", bootstyle = "info", command = self._view3D, state = DISABLED)
        self.view3DButt.pack(side = "left", padx = 5)

        # Add a progress bar
        self.progressBar = tb.Progressbar(self.topSectionFrame, mode = "determinate", length = 400)


    def _resetUI(self):
        self.filePath.set("")
        self.resultsDf = pd.DataFrame()

        self.runButt.state([DISABLED])
        self.saveButt.state([DISABLED])
        self.view3DButt.state([DISABLED])

        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()
            self.canvas = None
            self.fig = None
            self.ax = None

        if self.plotFrame:
            self.plotFrame.pack_forget()
            self.plotFrame.destroy()
            self.plotFrame = None

        # Reset the layout (keep topSectionFrame intact)
        self.topSectionFrame.pack_forget()
        self.topSectionFrame.pack(pady = (100, 0))


    def _browse(self):
        path = filedialog.askopenfilename(title = "Select an OEB file", initialdir = os.getcwd())

        if not path:
            return

        if path == self.filePath.get():
            return

        # Validate the input file extension
        if path.lower().endswith((".oeb", ".oeb.gz")):
            self.filePath.set(path)

            self.view3DButt.state([DISABLED])
            self.saveButt.state([DISABLED])
            self.runButt.state(["!disabled"])
        else:
            messagebox.showerror("Invalid File", "Please select a file ending in .oeb or .oeb.gz", parent = self)


    def _run(self):
        path = self.filePath.get()

        # Prepare the progress bar
        totalSteps = 4
        self.progressBar["maximum"] = totalSteps
        self.progressBar["value"] = 0
        self.progressBar.pack(pady = 10)
        self.update()

        if not checkContent(path):
            self.progressBar.pack_forget()
            messagebox.showerror("Invalid File", "The selected file is empty.", parent = self)
            
            self._resetUI()
            self.update_idletasks()
            return
        
        # Advance the progress bar by one step
        self.progressBar.step(1)
        self.update()

        df = constructDataFrame(path)

        self.progressBar.step(1)
        self.update()

        if not checkKeywords(df):
            self.progressBar.pack_forget()
            messagebox.showerror("Invalid File", "The selected file does not contain the required keywords.", parent = self)
            
            self._resetUI()
            self.update_idletasks()
            return
        
        self.progressBar.step(1)
        self.update()

        results = processAndClusterData(df)

        self.progressBar.step(1)
        self.update()

        if results is None:
            self.progressBar.pack_forget()
            messagebox.showerror("Clustering Error", "No valid clustering results found.", parent = self)
            
            self._resetUI()
            self.update_idletasks()
            return

        xPca, labels, resultsDf = results
        self.resultsDf = resultsDf
        filteredLabels = set(labels) - {-1}

        if self.plotFrame:
            self.plotFrame.pack_forget()
            self.plotFrame.destroy()

        self.plotFrame = tb.Frame(self)
        self.plotFrame.pack(fill = "both", expand = True)

        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()
            self.canvas = None
            self.fig = None
            self.ax = None

        # Compute the figure size as 90% of plotFrame’s width and 80% of its height
        self.plotFrame.update_idletasks()

        dpi = 100
        figWidth = (self.plotFrame.winfo_width() * 0.9) / dpi
        figHeight = (self.plotFrame.winfo_height() * 0.8) / dpi

        self.fig, self.ax = plt.subplots(figsize = (figWidth, figHeight), dpi = dpi)
        self.canvas = FigureCanvasTkAgg(self.fig, master = self.plotFrame)
        self.canvas.get_tk_widget().pack(fill = "both", expand = True)

        self.ax.clear()

        for id in filteredLabels:
            mask = (labels == id)
            self.ax.scatter(xPca[mask, 0], xPca[mask, 1], label = "Cluster " + str(id), s = 50)

        self.ax.set_title("DBSCAN Clustering")
        self.ax.set_xlabel("Principal Component 1")
        self.ax.set_ylabel("Principal Component 2")
        self.ax.legend(bbox_to_anchor = (1.02, 1), loc = "upper left", borderaxespad = 0)

        self.fig.subplots_adjust(left = 0.1, right = 0.9, top = 0.9, bottom = 0.15)
        self.canvas.draw()

        self.progressBar.pack_forget()

        self.saveButt.state(["!disabled"])
        self.view3DButt.state(["!disabled"])
    

    def _view3D(self):
        if self.resultsDf.empty:
            messagebox.showwarning("No Clusters", "Please run clustering first.", parent = self)
            return

        # Create a popup window
        popup = tk.Toplevel(self)
        popup.title("3D Molecule Viewer")
        popup.geometry("400x400")

        # Prepare one selectable list for clusters and another for molecules
        clusters = sorted(self.resultsDf["cluster label"].unique())
        clusterVar = tk.StringVar(value = str(clusters[0]))
        molVar = tk.StringVar()

        def updateMolList(*args):
            # Update the molecule list based on the selected cluster
            clusterId = int(clusterVar.get())
            mols = self.resultsDf[self.resultsDf["cluster label"] == clusterId]["title"].tolist()
            molMenu["menu"].delete(0, "end")

            for mol in mols:
                molMenu["menu"].add_command(label = mol, command = lambda value = mol: molVar.set(value))

            if mols:
                molVar.set(mols[0])

        # Add labels and option menus for clusters and molecules
        tk.Label(popup, text = "Cluster:").pack(pady = (10, 0))
        tk.OptionMenu(popup, clusterVar, *clusters, command = updateMolList).pack()

        tk.Label(popup, text = "Molecule:").pack(pady = (10, 0))
        molMenu = tk.OptionMenu(popup, molVar, "")
        molMenu.pack()

        updateMolList()

        def show3D():
            # Render and open the 3D structure of the selected molecule in a web browser
            title = molVar.get()
            smiles = self.resultsDf[self.resultsDf["title"] == title]["SMILES"].values[0]
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.UFFOptimizeMolecule(mol)
            mb = Chem.MolToMolBlock(mol)

            view = py3Dmol.view(width = 600, height = 400)
            view.addModel(mb, "mol")
            view.setStyle({"stick": {}})
            view.zoomTo()
            html = view._make_html()

            # Create a temporary HTML file to display the 3D structure
            with tempfile.NamedTemporaryFile("w", delete = False, suffix = ".html") as f:
                f.write(html)
                tempPath = f.name

            webbrowser.open("file://" + str(tempPath))

        # Add a button to open the 3D viewer
        tb.Button(popup, text = "Open 3D Viewer", bootstyle = "primary", command = show3D).pack(pady = 10)


    def _save(self):
        path = filedialog.asksaveasfilename(parent = self, title = "Save Clustering Results", defaultextension = ".csv", filetypes = [("CSV files", "*.csv")])

        if not path:
            return

        try:
            self.resultsDf.to_csv(path, index = False)
            messagebox.showinfo(title = "File Saved", message = "Results successfully saved to:\n" + str(path), parent = self)
        except Exception as e:
            messagebox.showerror(title = "Save Error", message = "Could not save file:\n" + str(e), parent = self)
            
            self._resetUI()
            self.update_idletasks()


if __name__ == "__main__":
    app = ClusteringApp()
    app.mainloop()