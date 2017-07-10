function [GRAD_MATRIX,IMAGE_NUM] = img_grad(IMAGE,IMAGE1)


grad_x_kernel=[-1 0 1;-2 0 2;-1 0 1];
grad_y_kernel=[1 2 1;0 0 0;-1 -2 -1];
IMAGE_NUM=1;

 for iFile = 1:size(IMAGE1,1);
     
     A=imread([IMAGE IMAGE1(iFile).name]); 
     A_1=A(:,:,1);
     A_2=A(:,:,2);
     A_3=A(:,:,3);
     
     grad_x_1=conv2(double(A_1),double(grad_x_kernel),'same');
     grad_y_1=conv2(double(A_1),double(grad_y_kernel),'same');
     
     grad_x_2=conv2(double(A_2),double(grad_x_kernel),'same');
     grad_y_2=conv2(double(A_2),double(grad_y_kernel),'same');
     
     grad_x_3=conv2(double(A_3),double(grad_x_kernel),'same');
     grad_y_3=conv2(double(A_3),double(grad_y_kernel),'same');
     
     
     grad_mag_1=sqrt(((grad_x_1.^2)+(grad_y_1.^2)));
     grad_mag_2=sqrt(((grad_x_2.^2)+(grad_y_2.^2)));
     grad_mag_3=sqrt(((grad_x_3.^2)+(grad_y_3.^2)));
     
     grad_mag=sqrt(((grad_mag_1.^2)+(grad_mag_2.^2)+(grad_mag_3.^2)));
     grad_mag = uint8(round(grad_mag));
     %grad_mag=reshape(grad_mag,[40 30 1]);
    
     grad_dir_1=(atan2d(grad_y_1,grad_x_1));
     grad_dir_2=(atan2d(grad_y_2,grad_x_2));
     grad_dir_3=(atan2d(grad_y_3,grad_x_3));
     
     grad_dir=sqrt(((grad_dir_1.^2)+(grad_dir_2.^2)+(grad_dir_3.^2)));
     grad_dir = uint8(round(grad_dir));
     %grad_dir=reshape(grad_dir,[40 30 1]);
    
     grad_matrix=zeros(40,30,2);
     grad_matrix(:,:,1)=grad_mag;
     grad_matrix(:,:,2)=grad_dir;
     grad_matrix=reshape(grad_matrix,[2400 1]);
     
     GRAD_MATRIX(:,IMAGE_NUM)=grad_matrix;
     
     %grad_mag_1 = uint8(round(grad_mag_1*255));
     %grad_mag_1=reshape(grad_mag_1,[40 30 1]);
     %grad_mag_2 = uint8(round(grad_mag_2*255));
     %grad_mag_2=reshape(grad_mag_2,[40 30 1]);
     %grad_mag_3 = uint8(round(grad_mag_3*255));
     %grad_mag_3=reshape(grad_mag_3,[40 30 1]);
     
     %grad_dir_1 = uint8(round(grad_dir_1*255));
     %grad_dir_1=reshape(grad_dir_1,[40 30 1]);
     %grad_dir_2 = uint8(round(grad_dir_2*255));
     %grad_dir_2=reshape(grad_dir_2,[40 30 1]);
     %grad_dir_3 = uint8(round(grad_dir_3*255));
     %grad_dir_3=reshape(grad_dir_3,[40 30 1]);
     
     
     %grad_mag(:,:,1)=grad_mag_1;
     %grad_mag(:,:,2)=grad_mag_2;
     %grad_mag(:,:,3)=grad_mag_3;
     
     
     %grad_dir(:,:,1)=grad_dir_1;
     %grad_dir(:,:,2)=grad_dir_2;
     %grad_dir(:,:,3)=grad_dir_3;
     
     %grad_mag=rgb2gray(grad_mag);
     %figure;
     %showIm=grad_mag;
     %imshow(showIm);
     
     %grad_dir=rgb2gray(grad_dir);
     %figure;
     %showIm=grad_dir;
     %imshow(showIm);
     IMAGE_NUM=IMAGE_NUM+1;
     
 end
 