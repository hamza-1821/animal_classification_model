
It is preferrable that you work in a virtual space(reccomended).
It's fine if you work in your system.

Make sure you have following dependencies installed in your system
librosa,numpy,pandas,scikit-learn,matplotlib and joblib
"pip3 install librosa numpy pandas scikit-learn matplotlib joblib"
librosa generates a bunch of warnings, because it fails in using pysound, uses audioread instead.

(optional)
the .sh file is a script that deals with format outliers of the data set. Before running the .py file, execute the .sh script by typing 
chmod x <file_name>
./<file_name>
