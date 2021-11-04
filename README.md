# Wavelet Search

## Installation

Please ensure that you have Python 3.9 installed on your computer and configured
in your PATH before continuing. Although the application should run with later
versions of Python, it has only been tested on Python 3.9.7. You can check your
current Python version with the following command:

```
python3 --version
```

To set up your Python environment to run this application, simply open a 
terminal window in the root directory of the this project (i.e. where this 
README.md is located) and run the following command:

```
make setup
```

This will install all the required package dependencies for this application, 
including:

 - numpy 1.21.3
 - opencv-python 4.5.4.58
 - pywavelets 1.1.1
 - matplotlib 3.4.3
 - pysimplegui 4.53.0

## Running the Application

Once you have followed the installation steps above, simply run the following
command to launch the application:

```
make run
```