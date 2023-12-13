#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import torch

def Interpolation_Coefficient(K):
    
    Coefficient = np.zeros((K*K,K*K))
    
    if (K==3):
        
        ##############################################################
        #(3x3 conv)--->9*9
        
        #0
        Coefficient[0,0]=(1/2)
        Coefficient[0,1]=(np.sqrt(2)/2-1/2)
        Coefficient[0,3]=(np.sqrt(2)/2-1/2)
        Coefficient[0,4]=(3/2-np.sqrt(2))

        #1
        Coefficient[1,1]=1.0

        #2
        Coefficient[2,1]=(np.sqrt(2)/2-1/2)
        Coefficient[2,2]=(1/2)
        Coefficient[2,4]=(3/2-np.sqrt(2))
        Coefficient[2,5]=(np.sqrt(2)/2-1/2)

        #3
        Coefficient[3,3]=1.0

        #4
        Coefficient[4,4]=1.0

        #5
        Coefficient[5,5]=1.0

        #6
        Coefficient[6,3]=(np.sqrt(2)/2-1/2) 
        Coefficient[6,4]=(3/2-np.sqrt(2))
        Coefficient[6,6]=(1/2)
        Coefficient[6,7]=(np.sqrt(2)/2-1/2)

        #7
        Coefficient[7,7]=1.0

        #8
        Coefficient[8,4]=(3/2-np.sqrt(2)) 
        Coefficient[8,5]=(np.sqrt(2)/2-1/2) 
        Coefficient[8,7]=(np.sqrt(2)/2-1/2) 
        Coefficient[8,8]=(1/2)

    
    if (K==5):
        
        ##############################################################
        #(5x5 conv)--->25*25
    
        #0
        Coefficient[0,0]=0.171573 
        Coefficient[0,1]=0.242641
        Coefficient[0,5]=0.242641   
        Coefficient[0,6]=0.343146 

        #1
        Coefficient[1,1]=0.64872
        Coefficient[1,2]=0.19928
        Coefficient[1,6]=0.11628 
        Coefficient[1,7]=0.03572

        #2
        Coefficient[2,2]=1.0

        #3
        Coefficient[3,2]=0.496928
        Coefficient[3,3]=0.351072
        Coefficient[3,7]=0.089072
        Coefficient[3,8]=0.062928

        #4
        Coefficient[4,3]=0.242641
        Coefficient[4,4]=0.171573
        Coefficient[4,8]=0.343146
        Coefficient[4,9]=0.242641                      

        #5
        Coefficient[5,5]=0.351072 
        Coefficient[5,6]=0.062928
        Coefficient[5,10]=0.496928 
        Coefficient[5,11]=0.089072 

        #6
        Coefficient[6,6]=(1/2) 
        Coefficient[6,7]=(np.sqrt(2)/2-1/2) 
        Coefficient[6,11]=(np.sqrt(2)/2-1/2) 
        Coefficient[6,12]=(3/2-np.sqrt(2))

        #7
        Coefficient[7,7]=1.0

        #8
        Coefficient[8,7]=(np.sqrt(2)/2-1/2)
        Coefficient[8,8]=(1/2) 
        Coefficient[8,12]=(3/2-np.sqrt(2)) 
        Coefficient[8,13]=(np.sqrt(2)/2-1/2)

        #9
        Coefficient[9,8]=0.11628 
        Coefficient[9,9]=0.64872 
        Coefficient[9,13]=0.03572 
        Coefficient[9,14]=0.19928 

        #10
        Coefficient[10,10]=1.0 

        #11
        Coefficient[11,11]=1.0 

        #12
        Coefficient[12,12]=1.0 

        #13
        Coefficient[13,13]=1.0 

        #14
        Coefficient[14,14]=1.0 

        #15
        Coefficient[15,10]=0.19928
        Coefficient[15,11]=0.03572 
        Coefficient[15,15]=0.64872
        Coefficient[15,16]=0.11628 

        #16
        Coefficient[16,11]=(np.sqrt(2)/2-1/2)
        Coefficient[16,12]=(3/2-np.sqrt(2))
        Coefficient[16,16]=(1/2)
        Coefficient[16,17]=(np.sqrt(2)/2-1/2) 

        #17
        Coefficient[17,17]=1.0

        #18
        Coefficient[18,12]=(3/2-np.sqrt(2)) 
        Coefficient[18,13]=(np.sqrt(2)/2-1/2) 
        Coefficient[18,17]=(np.sqrt(2)/2-1/2) 
        Coefficient[18,18]=(1/2)

        #19
        Coefficient[19,13]=0.089072 
        Coefficient[19,14]=0.496928 
        Coefficient[19,18]=0.062928
        Coefficient[19,19]=0.351072 

        #20
        Coefficient[20,15]=0.242641
        Coefficient[20,16]=0.343146
        Coefficient[20,21]=0.242641 
        Coefficient[20,20]=0.171573

        #21
        Coefficient[21,16]=0.062928
        Coefficient[21,17]=0.089072 
        Coefficient[21,21]=0.351072 
        Coefficient[21,22]=0.496928  

        #22
        Coefficient[22,22]=1.0

        #23
        Coefficient[23,17]=0.03572
        Coefficient[23,18]=0.11628  
        Coefficient[23,22]=0.19928
        Coefficient[23,23]=0.64872 

        #24
        Coefficient[24,18]=0.343146 
        Coefficient[24,19]=0.242641 
        Coefficient[24,23]=0.242641 
        Coefficient[24,24]=0.171573
    
    
    if (K==7):
        
        ##############################################################
        #(7x7 conv)--->49*49
    
        #0
        Coefficient[0,0]=0.106602  
        Coefficient[0,1]=0.014719  
        Coefficient[0,7]=0.7720780
        Coefficient[0,8]=0.106602

        #1
        Coefficient[1,1]=0.299038
        Coefficient[1,2]=0.299038  
        Coefficient[1,8]=0.200962
        Coefficient[1,9]=0.200962

        #2
        Coefficient[2,2]=0.697086
        Coefficient[2,3]=0.200692
        Coefficient[2,9]=0.079371
        Coefficient[2,10]=0.022851

        #3
        Coefficient[3,3]=1.0

        #4
        Coefficient[4,3]=0.200692
        Coefficient[4,4]=0.697086
        Coefficient[4,10]=0.022851
        Coefficient[4,11]=0.079371

        #5
        Coefficient[5,4]=0.299038
        Coefficient[5,5]=0.299038
        Coefficient[5,11]=0.200962
        Coefficient[5,12]=0.200962

        #6
        Coefficient[6,5]=0.7720780
        Coefficient[6,6]=0.106602
        Coefficient[6,12]=0.106602
        Coefficient[6,13]=0.014719

        #7
        Coefficient[7,7]=0.299038
        Coefficient[7,8]=0.200962
        Coefficient[7,14]=0.299038
        Coefficient[7,15]=0.200962

        #8
        Coefficient[8,8]=0.171573
        Coefficient[8,9]=0.242641
        Coefficient[8,15]=0.242641  
        Coefficient[8,16]=0.343146

        #9
        Coefficient[9,9]=0.64872
        Coefficient[9,10]=0.19928
        Coefficient[9,16]=0.11628
        Coefficient[9,17]=0.03572

        #10
        Coefficient[10,10]=1.0

        #11
        Coefficient[11,10]=0.496928
        Coefficient[11,11]=0.351072
        Coefficient[11,17]=0.089072
        Coefficient[11,18]=0.062928

        #12
        Coefficient[12,11]=0.242641
        Coefficient[12,12]=0.171573
        Coefficient[12,18]=0.343146
        Coefficient[12,19]=0.242641 

        #13
        Coefficient[13,12]=0.200962
        Coefficient[13,13]=0.299038
        Coefficient[13,19]=0.200962 
        Coefficient[13,20]=0.299038 

        #14
        Coefficient[14,14]=0.697086
        Coefficient[14,15]=0.079371
        Coefficient[14,21]=0.200692 
        Coefficient[14,22]=0.022851

        #15
        Coefficient[15,15]=0.351072
        Coefficient[15,16]=0.062928 
        Coefficient[15,22]=0.496928  
        Coefficient[15,23]=0.089072 

        #16 
        Coefficient[16,16]=(1/2) 
        Coefficient[16,17]=(np.sqrt(2)/2-1/2) 
        Coefficient[16,23]=(np.sqrt(2)/2-1/2) 
        Coefficient[16,24]=(3/2-np.sqrt(2)) 

        #17
        Coefficient[17,17]=1.0

        #18
        Coefficient[18,17]=(np.sqrt(2)/2-1/2) 
        Coefficient[18,18]=(1/2)
        Coefficient[18,24]=(3/2-np.sqrt(2)) 
        Coefficient[18,25]=(np.sqrt(2)/2-1/2)

        #19
        Coefficient[19,18]=0.11628 
        Coefficient[19,19]=0.64872 
        Coefficient[19,25]=0.03572 
        Coefficient[19,26]=0.19928 

        #20
        Coefficient[20,19]=0.079371 
        Coefficient[20,20]=0.697086 
        Coefficient[20,26]=0.022851 
        Coefficient[20,27]=0.200692   

        #21
        Coefficient[21,21]=1.0

        #22
        Coefficient[22,22]=1.0

        #23
        Coefficient[23,23]=1.0

        #24
        Coefficient[24,24]=1.0

        #25
        Coefficient[25,25]=1.0

        #26
        Coefficient[26,26]=1.0

        #27
        Coefficient[27,27]=1.0

        #28
        Coefficient[28,21]=0.200692 
        Coefficient[28,22]=0.022851 
        Coefficient[28,28]=0.697086 
        Coefficient[28,29]=0.079371

        #29
        Coefficient[29,22]=0.19928 
        Coefficient[29,23]=0.03572 
        Coefficient[29,29]=0.64872 
        Coefficient[29,30]=0.11628

        #30
        Coefficient[30,23]=(np.sqrt(2)/2-1/2)
        Coefficient[30,24]=(3/2-np.sqrt(2)) 
        Coefficient[30,30]=(1/2) 
        Coefficient[30,31]=(np.sqrt(2)/2-1/2) 

        #31
        Coefficient[31,31]=1.0

        #32
        Coefficient[32,24]=(3/2-np.sqrt(2)) 
        Coefficient[32,25]=(np.sqrt(2)/2-1/2) 
        Coefficient[32,31]=(np.sqrt(2)/2-1/2) 
        Coefficient[32,32]=(1/2) 

        #33
        Coefficient[33,25]=0.089072 
        Coefficient[33,26]=0.496928 
        Coefficient[33,32]=0.062928 
        Coefficient[33,33]=0.351072

        #34
        Coefficient[34,26]=0.022851 
        Coefficient[34,27]=0.200692 
        Coefficient[34,33]=0.079371 
        Coefficient[34,34]=0.697086

        #35 
        Coefficient[35,28]=0.299038
        Coefficient[35,29]=0.200962
        Coefficient[35,35]=0.299038 
        Coefficient[35,36]=0.200962

        #36            
        Coefficient[36,29]=0.242641 
        Coefficient[36,30]=0.343146 
        Coefficient[36,36]=0.171573 
        Coefficient[36,37]=0.242641 

        #37
        Coefficient[37,30]=0.062928 
        Coefficient[37,31]=0.089072 
        Coefficient[37,37]=0.351072 
        Coefficient[37,38]=0.496928 

        #38
        Coefficient[38,38]=1.0

        #39
        Coefficient[39,31]=0.03572
        Coefficient[39,32]=0.11628
        Coefficient[39,38]=0.19928
        Coefficient[39,39]=0.64872

        #40
        Coefficient[40,32]=0.343146 
        Coefficient[40,33]=0.242641 
        Coefficient[40,39]=0.242641
        Coefficient[40,40]=0.171573 

        #41
        Coefficient[41,33]=0.200962 
        Coefficient[41,34]=0.299038 
        Coefficient[41,40]=0.200962 
        Coefficient[41,41]=0.299038 

        #42
        Coefficient[42,35]=0.014719 
        Coefficient[42,36]=0.106602 
        Coefficient[42,42]=0.106602 
        Coefficient[42,43]=0.7720780

        #43
        Coefficient[43,36]=0.200962 
        Coefficient[43,37]=0.200962 
        Coefficient[43,43]=0.299038 
        Coefficient[43,44]=0.299038

        #44
        Coefficient[44,37]=0.079371 
        Coefficient[44,38]=0.022851 
        Coefficient[44,44]=0.697086 
        Coefficient[44,45]=0.200692

        #45
        Coefficient[45,45]=1.0

        #46
        Coefficient[46,38]=0.022851 
        Coefficient[46,39]=0.079371 
        Coefficient[46,45]=0.200692 
        Coefficient[46,46]=0.697086

        #47
        Coefficient[47,39]=0.200962
        Coefficient[47,40]=0.200962 
        Coefficient[47,46]=0.299038 
        Coefficient[47,47]=0.299038 

        #48
        Coefficient[48,40]=0.106602
        Coefficient[48,41]=0.7720780
        Coefficient[48,47]=0.014719 
        Coefficient[48,48]=0.106602 

    
    Coefficient = torch.from_numpy(Coefficient).float()
    
    return Coefficient 