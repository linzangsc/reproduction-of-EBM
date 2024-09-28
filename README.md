## Reproduction of Energy-based Model with Contrastive Divergence and Langevin Dynamics by Pytorch

the code of utils.py/Sampler refers to https://github.com/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/tutorial8/Deep_Energy_Models.ipynb

usage: python train.py

some sampling results after training 100 epochs:

![imageData](https://github.com/user-attachments/assets/a4805a5c-9024-4d74-b8b2-06f1144ef57e)

loss trend: 

energy of real samples are slightly lower than fake samples

![image](https://github.com/user-attachments/assets/affbd5db-0ba1-410c-a953-b405536c9266)

contrastive divergence keeps fluctuating around 0 while regularization loss converges to 0 soon

![image](https://github.com/user-attachments/assets/2e18e2fa-c46f-46aa-8f91-37fe113adf7d)
