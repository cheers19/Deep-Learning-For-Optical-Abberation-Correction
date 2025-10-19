# Deep-Learning-For-Optical-Abberation-Correction

This work focuses on analyzing optical system data and developing a deep learning model to accurately classify and correct optical aberrations.

## Data Structure and Features

The dataset is structured to link system parameters to specific aberration types:

| Column Group | Description | Examples |
| :--- | :--- | :--- |
| **Target Columns** | Aberration error types and their respective numeric error value. | `Tilt_E1`, `Decenter_E3`, *Numeric Value* |
| **Feature Columns** | System parameters. | Wavefront Aberration Coefficients (Zernike C2,...,Zernike C37) and MTF score|

 
**Note:** The *aberration error type* is composed of the **aberration type** and the **element name**, that is, for `Tilt_E1` the name 'Tilt' is the aberration type (which we denote as 'error_name') and the *element*, 'E1' (which we denote as 'parameter_name').


### Exploratory Data Analysis (EDA) Insights

Initial analysis revealed a crucial sparsity and dependency in the feature space:

* For any given aberration type (e.g., 'Tilt', 'Decenter', etc.), **only a few aberration coefficients significantly contribute** to the prediction. Each unique aberration type is charecterised with a unique coefficients distribution.
* The **relative fraction** (or importance) of this distribution **changes dynamically** with the numeric error value of the aberration type.
* The distribution is independent of the element name (e.g., E1, E2, E3, etc.)

This is presented in Fig. 1: 

<div align="center">

  <img 
    width="700" 
    src="https://github.com/user-attachments/assets/7456b6d2-3101-435b-830e-9582a1d255eb" 
    alt="Fig_1" 
    style="display: block; margin: 0 auto; max-width: 100%; height: auto;"
  />

  <br>**Fig. 1: The wavefront aberration coefficients for different aberration types. left and right plots correspond to error values in the lower and upper quartile**
  <br>
  <br>

</div>

---

# Model Architectures 
## Simple fully connected deep neural network
A simple, fully connected deep neural network (see Fig. 2) **failed to accurately classify** the aberration error types (see Fig. 3). 

<div align="center">

  <img 
    width="300" 
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
    src="https://github.com/user-attachments/assets/56a29cda-892c-4267-8c0a-e01436ae3cdb" 
    alt="classification with MLP without 2 stage model" 
    style="display: block; margin: 0 auto; max-width: 100%; height: auto;"
  />

  <br>**Fig. 3: A simple, fully connected deep neural network fails to predict the aberration error types**
  <br>
  <br>

</div>

This challenge was addressed by implementing a **cascading classification architecture** that breaks the prediction into two *dependent* steps:
## cascade Classification
### **Aberration Type Naming Convention**

As mentioned, each aberration error type is composed of 'error_name'+'parameter_name'. 
Since every element ('parameter_name') can only experience certain types of aberrations ('error_name'), **it make sense to use a cascading classification model where the prediction of the aberration type preceds the element classification**.


### **Casceding Model Implementation**

The final model (see Fig. 3) uses a cascading approach to improve classification accuracy:

1.  **Error type Identification:** A layer recives the information learned by the two inputs and identifies the categorical value for ***`error_name`***.
2.  **Element Identification:** The outputs of the `error_name` layer are **concatenated** with the initial input features. This combined output then serves as the input for another layer, which identifies the categorical value of ***`parameter_name`*** (the optical component).

The results of this casceding classification model are presented in Fig. 5:

<div align="center">

  <img 
    width="500" 
    src="https://github.com/user-attachments/assets/07829e80-eee6-4b21-aaab-d449065ab6ae" 
    alt="neural_network_architecture_cascading" 
    style="display: block; margin: 0 auto; max-width: 100%; height: auto;"
  />

  <br>**Fig. 3: Cascade classification deep neural network**
  <br>
  <br>

</div>

<div align="center">

  <img 
    width="500" 
    src="https://github.com/user-attachments/assets/2db76324-b98e-4d18-9f01-54133969e4c7" 
    alt="error_type_casceding_OneInputLayer" 
    style="display: block; margin: 0 auto; max-width: 100%; height: auto;"
  />

  <br>**Fig. 4: Aberration error type prediction using the cascading calssification model**
  <br>
  <br>

</div>

<div align="center">

  <img 
    width="500" 
    src="https://github.com/user-attachments/assets/3bc368f7-1d12-40f7-b4af-95e9630cd4ed" 
    alt="element_type_casceding_OneInputLayer" 
    style="display: block; margin: 0 auto; max-width: 100%; height: auto;"
  />

  <br>**Fig. 5: Optical element prediction using the cascading calssification model**
  <br>
  <br>

</div>

**Further data need to be collected, as well as mixed error types to achieve a better classification accuracy**

---

## Numerical error prediction

Once the classification (both `error_name` and `parameter_name`) is achieved, a **simple linear regression model** is applied specifically to rows labeled with the predicted classes. This post-classification regression demonstrated *extremely high accuracy* in determining the required correction value.

This two-step process—casceding classification followed by focused regression—successfully addresses the data's inherent complexity and sparsity.
