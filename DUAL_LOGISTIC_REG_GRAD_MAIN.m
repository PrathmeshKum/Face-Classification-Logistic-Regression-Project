% Dual Logistic Regression For Gradient Domain

%@Zhaozheng Yin, spring 2017

clc; clear all; %close all;
dir_training1 = 'trainingImages\face_resized\';
dir_training2 = 'trainingImages\background_resized\';
dir_testing1 = 'testingImages\face_resized\';
dir_testing2 = 'testingImages\background_resized\';

tt=cputime;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
files = dir([dir_training1 '*.jpg']);
files1 = dir([dir_training2 '*.jpg']);

[X1,train_img_num1]=img_grad(dir_training1,files);
[X2,train_img_num2]=img_grad(dir_training2,files1);

X = [X1, X2]; w = [];
train_img_num=size(files,1)+size(files1,1);


for i = 1:train_img_num1
    
    w = [w; 1];
    
end

for i = 1:train_img_num2
   
    w = [w; 0];
  
end

X = double(X); %every column in X is one vectorized input image
X = [ones(1,size(X,2)); X];
%X=normalizeIm1(X,train_img_num); % Normalizing

initial_psi=zeros(train_img_num,1); % Initializing (I*1) matrix
%initial_phi=zeros(241,1);
%phi=ones(3601,1);

sig=sum(var(X));    % not required in this model
var_prior=sig*10;

disp('Training Image Matrix Created ');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%testing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
files = dir([dir_testing1 '*.jpg']);
files1 = dir([dir_testing2 '*.jpg']);


[X_test1,test_img_num1]=img_grad(dir_testing1,files);
[X_test2,test_img_num2]=img_grad(dir_testing2,files1);

X_test = [X_test1, X_test2]; w_test = [];
test_img_num=size(files,1)+size(files1,1);


for i = 1:test_img_num1
    
    w_test = [w_test; 1];
    
end

for i = 1:test_img_num2
   
    w_test = [w_test; 0];
  
end

X_test = double(X_test); %every column in X is one vectorized input image
X_test = [ones(1,size(X_test,2)); X_test];
%X_test=normalizeIm1(X_test,test_img_num); % Normalizing

disp('Testing Image Matrix Created ');

[predictions, phi] = fit_dlogr (X, w, var_prior, X_test, initial_psi);


% INFERENCE

disp('Computing Inference');

predictions=abs(predictions);

for i = 1:test_img_num;
    
    A=vpa(predictions(1,i),10);
    A=round(A);
    if A==1;  % Decision Boundary of 0.5
        
        y_cap(1,i)=1;
        
    else
        
        y_cap(1,i)=0;
    
    end
end

w_test=w_test';
Miss_Detection_Num=0;
False_Alarm_Num =0;

for i = 1:test_img_num1;
    
    abs_error1(1,i)=predictions(1,i)-w_test(1,i);
    
    if y_cap(1,i)==0;
       
        Miss_Detection_Num = Miss_Detection_Num + 1;
        
    end
    
end

mod_error1=abs(abs_error1);
Miss_Detection=(sum(mod_error1)/test_img_num1);     
Miss_Detection_Num=(Miss_Detection_Num/test_img_num1)*100;   % Percent of wrongly detected face images

for i = (test_img_num1+1):test_img_num;
    
    if y_cap(1,i)==1;
       
        False_Alarm_Num = False_Alarm_Num + 1;
        
    end
    
    abs_error2(1,(i-test_img_num1))=predictions(1,i)-w_test(1,i);
end

mod_error2=abs(abs_error2);
False_Alarm=(sum(mod_error2)/test_img_num2);  
False_Alarm_Num=(False_Alarm_Num/test_img_num2)*100;      % Percent of wrongly detected background images

disp(['file execution time: ' num2str(cputime-tt)]);

% VISUALIZATION

%1.

plot((1:232),w_test(1:232),'*');
title('Plot For Miss Detection (Face)');
xlabel('Image Number');
ylabel('Image Class [0/1]');
hold on;
plot((1:232),y_cap(1:232),'o');
legend('GT CLASS','PREDICTED CLASS','location','northeast');

%2.

figure;
plot((233:796),w_test(233:796),'*');
title('Plot For False Alarm (Background)');
xlabel('Image Number');
ylabel('Image Class [0/1]');
hold on;
plot((233:796),y_cap(233:796),'o');
legend('GT CLASS','PREDICTED CLASS','location','northeast');


%3.

figure;
plot((1:796),predictions,'*')
title('Plot For Actual Predicted Class');
xlabel('Image Number');
ylabel('Image Class [0/1]');
legend('PREDICTED CLASS','location','northeast');
