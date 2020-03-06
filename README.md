# deepstreamce
This is an implementation of DeepStreamCE algorithm for detetcing concept evolution in deep neural networks with streaming methods

This demo of DeepStreamCE trains a VGG16 deep neural network on two classes (0 and 1 - airplane and automobile) from the CIFAR-10 dataset, with a concept evolution (class 6 - Frog) applied.  The concept evolution detection results are stored in *outlierresults.csv file in the 'output' directory.
 
Run traindnn.ipynb to create a DNN trained on 2 classes of the CIFAR-10 dataset.  This will be stored in the 'models' directory

Run deepstreamce.ipnyb to load the DNN, get the activations, reduce activations, setup activation analysis and process the datastream.

It uses parameters from a setup file located in the 'setup' directory.  There is also a paramdef.txt file which explains the parameters.

The autoencoder will be automatically produced if it is not already there. It will be stored in the 'models/reduce' directory.

The 'subprocess' directory needs to contain MoaGateway.jar

MoaGateway.jar can be produced by compiling the java code in the java directory (intellij).

Other data setups from CIFAR-10 can be run by: 
	> changing the 'classes' array in traindnn.ipynb
	> Creating a new setup file in the setup directory and changing parameters:
		○ DnnModelPath
		○ DataClasses
		○ DataDiscrepancyClass
	> Changing the setupFile name in deepstreamce.ipnyb

DeepStreamCE is compared to OpenMax (
A. Bendale, T. Boult “Towards Open Set Deep Networks” IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016 pdf).  OpenMax code was adapted from https://github.com/abhijitbendale/OSDN to work with keras with help from a wrapper https://github.com/aadeshnpn/OSDN.
The OpenMax code adapted for DeepStreamCE comparison is stored in the OpenMax directory.
Run start_openmax_compare.ipynb to start.  This is setup to run the data setup of 01-6 (airplane, automobile and frog).
Other data setups can be run by changing the 'setups' parameter to a number between 1 and 8. The data setups these map to are defined in the code.
OpenMax contains code used in the CVPR 2016 paper on "Towards Open Set Deep Networks". The software packages for OpenMax code calls LibMR functions.  The LibMR library is included for completeness.

The system setup to run this code is Unix, 64 CPUs, 416GB RAM, 4 x Tesla T4 CPUs, Tensorflow 2.0. For the Python code, Jupyter and Python 3 is used except for the OpenMax component which is Python 2.0. The Java code was produced in Intellij, utilising components from the MOA framework (
Bifet, A., Holmes, G., Kirkby, R., Pfahringer, B.: MOA: Massive Online Analysis. Journal
of Machine Learning Research 11(May), 1601{1604 (2010). URL http://www.jmlr.org/papers/v11/bifet10a.html)
and Py4j (Dagenais, B.: Py4j - A Bridge between Python and Java. URL https://www.py4j.org/) to link the Python code with Java code.
