# Deep-Learning-For-Optical-Abberation-Correction

This project focuses on analyzing optical system data and developing a deep learning model to accurately classify and correct optical aberrations.

## Data Structure and Features

The dataset is structured to link system parameters to specific aberration types:

| Column Group | Description | Examples |
| :--- | :--- | :--- |
| **Target Columns** | Aberration error types and their respective numeric error value. | `Tilt_E1`, `Decenter_E3`, *Numeric Value* |
| **Feature Columns** | System parameters. | Wavefront Aberration Coefficients and visibility|

### Exploratory Data Analysis (EDA) Insights

Initial analysis revealed a crucial sparsity and dependency in the feature space:

* For any given aberration type (e.g., Astigmatism, Coma, Spherical), **only a few aberration coefficients significantly contribute** to the prediction. Each unique aberration type is charecterised with a unique coefficients distribution.
* The **relative fraction** (or importance) of this distribution **changes dynamically** with the numeric error value of the aberration type.
* The distribution is independent of the element type (e.g., E1, E2, E3, etc.)

This is presented in Fig. 1: 

<div align="center">

  <img 
    width="700" 
    src="https://github.com/user-attachments/assets/7456b6d2-3101-435b-830e-9582a1d255eb" 
    alt="Fig_1" 
    style="display: block; margin: 0 auto; max-width: 100%; height: auto;"
  />

  <br>**Fig. 1: The wavefront aberration coefficients for different ‘ErrorType’. left and right plots correspond to error values in the lower and upper quartile**
  <br>
  <br>

</div>

---

## Model Architectures: 
# Simple fully connected deep neural network
A simple, fully connected deep neural network (see Fig. 2) **failed to accurately classify** the aberration types (see Fig. 3). 

<div align="center">

  <img 
    width="450" 
    src="https://github.com/user-attachments/assets/ecfeb197-6098-4f1b-b41e-6800d6e92328" 
    alt="old_model_architecture" 
    style="display: block; margin: 0 auto; max-width: 100%; height: auto;"
  />

  <br>**Fig. 2: A simple, fully connected deep neural network**
  <br>
  <br>

</div>

<div align="center">

  <img 
    width="700" 
    src="https://github.com/user-attachments/assets/0202e0a2-cf46-41f3-ba29-7c311ea5aad8" 
    alt="classification with MLP without 2 stage model" 
    style="display: block; margin: 0 auto; max-width: 100%; height: auto;"
  />

  <br>**Fig. 3: A simple, fully connected deep neural network fails to predict the aberration error types**
  <br>
  <br>

</div>

This challenge was addressed by implementing a **cascading classification architecture** that breaks the prediction into two *dependent* steps:
# cascade Classification
### **Aberration Type Naming Convention**

Each aberration in the target column is defined as `ErrorType_En`, where:
* `ErrorType`: Categorical value (e.g., `tilt`, `decenter`).
* `En`: Optical component index (e.g., `E1`, `E2`, `E3`).

### **Casceding Model Implementation**

The final model (see Fig. 3) uses a cascading approach to improve classification accuracy and separate learning of input with different physical meaning (like Zernike numbers and modulation transfer functions):

1. **Dividing the input data:** We separate the data for learning two physically different input 
2.  **ErrorType Identification:** A layer recives the information learned by the two inputs and identifies the categorical value for ***`ErrorType`***.
3.  **Element Identification:** The outputs of the `ErrorType` layer are **concatenated** with the initial input features. This combined output then serves as the input for another layer, which identifies the categorical value of ***`En`*** (the optical component).

The results of this casceding classification model are presented in Fig. 5:



---

## Correction Phase and High Accuracy

Once the classification (both `ErrorType` and `En`) is achieved, a **simple linear regression model** is applied specifically to rows labeled with the predicted classes. This post-classification regression demonstrated *extremely high accuracy* in determining the required correction value.

This two-step process—casceding classification followed by focused regression—successfully addresses the data's inherent complexity and sparsity.
