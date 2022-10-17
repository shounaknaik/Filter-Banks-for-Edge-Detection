#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Colab file can be found at:
	https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

# Code starts here:

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#This function is written for visualization purposes.
def min_max_scaling(image):

	image=(image-np.min(image))/np.ptp(image)

	return image



def main():

	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""

	sobel_filter=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
	gaussian_kernel=np.array([[0,0,1,2,1,0,0],[0,3,13,22,13,3,0],[1,13,59,97,59,13,1],[2,22,97,159,97,22,2],[1,13,59,97,59,13,1],
	[0,3,13,22,13,3,0],[0,0,1,2,1,0,0]],dtype=object)

	gaussian_kernel=gaussian_kernel.astype('uint8')

	convolved_filter=cv2.filter2D(src=gaussian_kernel,ddepth=-1,kernel=sobel_filter)
	# # print(convolved_filter.shape)

	# # Following Code understood from https://pyimagesearch.com/2021/01/20/opencv-rotate-image/

	h,w=convolved_filter.shape
	cX=h//2
	cY=w//2

	DoG_filter_bank=[]
	angles=np.linspace(0,360,num=12)

	for angle in angles:
		rotation_matrix = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
		filter_created=cv2.warpAffine(convolved_filter, rotation_matrix, (h, w)) #last parameter is the destination image size
		DoG_filter_bank.append(filter_created)

	# print(DoG_filter_bank)

	fig,ax=plt.subplots(2,6)
	row=0
	column=0
	for i,filter in enumerate(DoG_filter_bank):

		if i>5:
			row=1
		column=i%6
		ax[row, column].set_xticks([])
		ax[row,column].set_yticks([])
		ax[row, column].imshow(filter,cmap='gray')
		column+=1

	# fig.title('DoG Filter Bank')
	plt.savefig('DoG.png')
	



	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""

	
	
	print('Making all banks')

	def gaussian_element(sigma_x,x,y,sigma_diff=False):

		if sigma_diff==True:
			sigma_y=3*sigma_x

		else:
			sigma_y=sigma_x

		
		g = 1/(2*np.pi*sigma_x*sigma_y)*np.exp( -((x*x/2*sigma_x*sigma_x) + (y*y/2*sigma_y*sigma_y)))

		return g
	
	def generate_gaussian_matrix(n,sigma,sigma_diff=False):

		kernel_range=int(n/2)

		x=np.linspace(-kernel_range,kernel_range,n)
		y=np.linspace(-kernel_range,kernel_range,n)


		gaussian_matrix=[]
		for i in x:
			temp_row=[]
			for j in y:
				temp_row.append(gaussian_element(sigma,int(i),int(j),sigma_diff))

			gaussian_matrix.append(temp_row)

		return np.array(gaussian_matrix)


	LMS_scales=np.sqrt(2)**np.array([0,1,2])

	LMS_bank=[]
	LMS_bank_2nd_degree=[]

	for scale in LMS_scales:

		scale_wise_li=[]
		scale_wise_li_2nd_degree=[]

		gaussian_kernel=generate_gaussian_matrix(7,scale,sigma_diff=True)

		sobel_filter=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

		convolved_filter=cv2.filter2D(src=gaussian_kernel,ddepth=-1,kernel=sobel_filter)
		second_convolved_filter=cv2.filter2D(src=convolved_filter,ddepth=-1,kernel=sobel_filter)

		h,w=convolved_filter.shape
		cX=h//2
		cY=w//2
	
		angles=np.linspace(0,120,num=6)
		for angle in angles:
			rotation_matrix = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)

			filter_created=cv2.warpAffine(convolved_filter, rotation_matrix, (h, w)) #last parameter is the destination image size
			second_filter_created=cv2.warpAffine(second_convolved_filter,rotation_matrix,(h,w))

			scale_wise_li.append(filter_created)
			scale_wise_li_2nd_degree.append(second_filter_created)

		LMS_bank.append(scale_wise_li)
		LMS_bank_2nd_degree.append(scale_wise_li_2nd_degree)


	gaussian_bank_small=[]
	gaussian_scales=np.sqrt(2)**np.array([0,1,2,3])

	for scale in gaussian_scales:

		gaussian_kernel=generate_gaussian_matrix(7,scale,sigma_diff=False)
		gaussian_bank_small.append(gaussian_kernel)



	def LOG_element(sigma_x,x,y):


		# Formula adopted from https://academic.mu.edu/phys/matthysd/web226/Lab02.htm

		g = -(1/(np.pi*sigma_x**4))*(1-((x**2+y**2)/2*sigma_x**2))*np.exp( -(((x*x+y*y)/2*sigma_x**2)))

		return g
	
	def generate_LOG_matrix(n,sigma):

		kernel_range=int(n/2)

		x=np.linspace(-kernel_range,kernel_range,n)
		y=np.linspace(-kernel_range,kernel_range,n)


		gaussian_matrix=[]
		for i in x:
			temp_row=[]
			for j in y:
				temp_row.append(LOG_element(sigma,int(i),int(j)))

			gaussian_matrix.append(temp_row)

		return np.array(gaussian_matrix)


	LoG_small_bank=[]

	gaussian_scales=np.sqrt(2)**np.array([0,1,2,3])

	for scale in gaussian_scales:
		LoG_small_bank.append(generate_LOG_matrix(7,scale))

	for scale in gaussian_scales:
		LoG_small_bank.append(generate_LOG_matrix(7,3*scale))


	LMS=np.array(LMS_bank)
	LMS=np.concatenate((LMS,np.array(LMS_bank_2nd_degree)),axis=1)
	small_log_plus_gauss=np.concatenate((np.array(LoG_small_bank),np.array(gaussian_bank_small)))
	small_log_plus_gauss=np.expand_dims(small_log_plus_gauss,axis=0)
	LMS=np.concatenate((LMS,small_log_plus_gauss),axis=0)


	LMS_bank_flattened= [filter for row in LMS for filter in row]

	fig,ax=plt.subplots(4,12)
	row=0
	column=0
	for i,filter in enumerate(LMS_bank_flattened):

		
		column=i%12
		ax[row, column].set_xticks([])
		ax[row,column].set_yticks([])
		ax[row, column].imshow(min_max_scaling(filter),cmap='gray')
		column+=1

		if ((i+1)%12)==0:
			row+=1

	plt.savefig('LMS.png')

	
	######################################### LML #############################################################################

	LML_scales=np.sqrt(2)**np.array([1,2,3])

	LML_bank=[]
	LML_bank_2nd_degree=[]

	for scale in LML_scales:

		scale_wise_li=[]
		scale_wise_second_degree_li=[]

		gaussian_kernel=generate_gaussian_matrix(7,scale,sigma_diff=True)

		sobel_filter=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

		convolved_filter=cv2.filter2D(src=gaussian_kernel,ddepth=-1,kernel=sobel_filter)
		second_convolved_filter=cv2.filter2D(src=convolved_filter,ddepth=-1,kernel=sobel_filter)


		h,w=convolved_filter.shape
		cX=h//2
		cY=w//2

		angles=np.linspace(0,120,num=6)
		for angle in angles:

			rotation_matrix = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)

			filter_created=cv2.warpAffine(convolved_filter, rotation_matrix, (h, w)) #last parameter is the destination image size
			filter_created_2nd_degree=cv2.warpAffine(second_convolved_filter, rotation_matrix, (h, w)) #last parameter is the destination image size

			scale_wise_li.append(filter_created)
			scale_wise_second_degree_li.append(filter_created_2nd_degree)

		LML_bank.append(scale_wise_li)
		LML_bank_2nd_degree.append(scale_wise_second_degree_li)




	gaussian_bank_large=[]
	gaussian_scales=np.sqrt(2)**np.array([1,2,3,4])

	for scale in gaussian_scales:

		gaussian_kernel=generate_gaussian_matrix(7,scale,sigma_diff=False)
		gaussian_bank_large.append(gaussian_kernel)



	LoG_large_bank=[]

	gaussian_scales=np.sqrt(2)**np.array([1,2,3,4])

	for scale in gaussian_scales:
		LoG_large_bank.append(generate_LOG_matrix(7,scale))

	for scale in gaussian_scales:
		LoG_large_bank.append(generate_LOG_matrix(7,3*scale))


	LML=np.array(LML_bank)
	LML=np.concatenate((LML,np.array(LML_bank_2nd_degree)),axis=1)
	large_log_plus_gauss=np.concatenate((np.array(LoG_large_bank),np.array(gaussian_bank_large)))
	large_log_plus_gauss=np.expand_dims(large_log_plus_gauss,axis=0)
	LML=np.concatenate((LML,large_log_plus_gauss),axis=0)

	LML_bank_flattened= [filter for row in LML for filter in row]

	fig,ax=plt.subplots(4,12)
	row=0
	column=0
	for i,filter in enumerate(LML_bank_flattened):

		
		column=i%12
		ax[row, column].set_xticks([])
		ax[row,column].set_yticks([])
		ax[row, column].imshow(min_max_scaling(filter),cmap='gray')
		column+=1

		if ((i+1)%12)==0:
			row+=1

	plt.savefig('LML.png')


	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""

	def gabor(sigma, theta,gamma, Lambda=0.1, psi=1,n=15):

		kernel_size=(n-1)//2

	
		x=np.arange(-kernel_size,kernel_size+1)
		y=np.arange(-kernel_size,kernel_size+1)

		x,y=np.meshgrid(x,y)

		#Rotation and changing axis values

		x_theta=x*np.cos(theta)+y*np.sin(theta)
		y_theta=-x*np.sin(theta)+y*np.cos(theta)


		gabor=np.exp(-((x_theta**2+((gamma**2)*y_theta**2))/2*(sigma**2)))*(np.cos(2*np.pi/Lambda*x_theta +psi))

		return gabor


	def make_gabor_filter(sigma,theta,gamma):

		return gabor(sigma,theta,gamma)

	gabor_parameters=[[1.4,1.2,0.6],[1.2,1.25,0.5],[1.5,1.5,0.9],[1.8,1.3,0.8]]

	gabor_filter_bank=[]
	for param_li in gabor_parameters:

		sigma,theta,gamma=param_li[0],param_li[1],param_li[2]
		gabor_filter_bank.append(make_gabor_filter(sigma,theta,gamma))


	# print(len(gabor_filter_bank))


	fig,ax=plt.subplots(1,4)
	for i,filter in enumerate(gabor_filter_bank):

		
		ax[i].set_xticks([])
		ax[i].set_yticks([])
		ax[i].imshow(min_max_scaling(filter),cmap='gray')



	plt.savefig('Gabor.png')


	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""

	radius_li=[10,15,20]
	variations=4

	angles=np.linspace(20,360,variations)


	half_disc_bank=[]
	for radius in radius_li:


		half_disk_image=np.zeros((2*radius+1,2*radius+1))
		x,y=np.ogrid[-radius:radius+1,-radius:radius+1]

		mask=x*x+y*y<=radius**2
		half_mask=x>=0

		mask=np.logical_and(mask,half_mask)

		half_disk_image[mask]=1

		scale_wise_li=[]

		h,w=half_disk_image.shape[0],half_disk_image.shape[1]

		cX=h//2
		cY=w//2

		for angle in angles:

			rotation_matrix = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
			filter_created=cv2.warpAffine(half_disk_image, rotation_matrix, (h, w)) #last parameter is the destination image size

			opp_filter=np.flip(filter_created,1)

			scale_wise_li.append(filter_created)
			scale_wise_li.append(opp_filter)

		half_disc_bank.append(scale_wise_li)

	half_disc_bank_flattened=[filter for row in half_disc_bank for filter in row]

	fig,ax=plt.subplots(3,8)
	row=0
	column=0
	for i,filter in enumerate(half_disc_bank_flattened):

		
		column=i%8
		ax[row, column].set_xticks([])
		ax[row,column].set_yticks([])
		ax[row, column].imshow(min_max_scaling(filter),cmap='gray')
		column+=1

		if ((i+1)%8)==0:
			row+=1

	plt.savefig('HDMasks.png')

	##This function used for chi square calculations
	def chi_square_distance(filter_1,filter_2,bins,img):
		chi_distance=img*0
		for i in range(bins):
			temp=np.ma.masked_where(img==i,img)

			#converting boolean mask into integers
			temp=temp.mask.astype(np.int32)


			g = cv2.filter2D(temp,-1,filter_1)
			h = cv2.filter2D(temp,-1,filter_2)

			chi_square=np.nan_to_num(((g-h)**2)/(g+h))
			
			chi_distance=chi_distance+chi_square


			return 0.5*chi_distance


	for photo_index in range(10):

		print(f'Generating result for {photo_index+1}')



		"""
		Generate Texton Map
		Filter image using oriented gaussian filter bank
		"""
		image_path=f"../BSDS500/Images/{photo_index+1}.jpg"
		image_colour=cv2.imread(image_path)

		image_grayscale=cv2.imread(image_path,0)


		texton_map=[]

		for filter in LML_bank_flattened:
			#convolve and append
			convolved_filter=cv2.filter2D(src=image_grayscale,ddepth=-1,kernel=filter)
			texton_map.append(convolved_filter)

		for filter in gabor_filter_bank:
			#convolve and append
			convolved_filter=cv2.filter2D(src=image_grayscale,ddepth=-1,kernel=filter)
			texton_map.append(convolved_filter)

		for filter in DoG_filter_bank:
			#convolve and append

			convolved_filter=cv2.filter2D(src=image_grayscale,ddepth=-1,kernel=filter)
			texton_map.append(convolved_filter)

		
		texton_map=np.array(texton_map)


		texton_map=np.transpose(texton_map,(1,2,0))

		"""
		Generate texture ID's using K-means clustering
		Display texton map and save image as TextonMap_ImageName.png,
		use command "cv2.imwrite('...)"
		"""
		height,width,channels=texton_map.shape

		kmeans = KMeans(n_clusters = 64, random_state = 1)

		input=np.reshape(texton_map,((height*width),channels))
		labels=kmeans.fit_predict(input)
		# texton_map=kmeans.labels_

		texton_map=np.reshape(labels,(height,width))

		fig=plt.figure()
		plt.imshow(texton_map,cmap='gray')

		plt.savefig(f'./{photo_index+1}/TextonMap_{photo_index+1}.png')


		"""
		Generate Texton Gradient (Tg)
		Perform Chi-square calculation on Texton Map
		Display Tg and save image as Tg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""

		
		Tg=[]
		for i in range(len(half_disc_bank_flattened)-1):

			filter_1=half_disc_bank_flattened[i]
			filter_2=half_disc_bank_flattened[i+1]

			dist=chi_square_distance(filter_1,filter_2,64,texton_map)

			Tg.append(dist)

		Tg=np.array(Tg)

		Tg=np.mean(Tg,axis=0)

		fig=plt.figure()
		plt.imshow(min_max_scaling(Tg),cmap='gray')

		plt.savefig(f'./{photo_index+1}/Tg_{photo_index+1}.png')




		"""
		Generate Brightness Map
		Perform brightness binning 
		"""
		height,width=image_grayscale.shape
		kmeans = KMeans(n_clusters = 16, random_state = 1)
		input=np.reshape(image_grayscale,((height*width),1))
		labels=kmeans.fit_predict(input)

		brightness_map=np.reshape(labels,(height,width))


		"""
		Generate Brightness Gradient (Bg)
		Perform Chi-square calculation on Brightness Map
		Display Bg and save image as Bg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""

		Bg=[]
		for i in range(len(half_disc_bank_flattened)-1):

			filter_1=half_disc_bank_flattened[i]
			filter_2=half_disc_bank_flattened[i+1]

			dist=chi_square_distance(filter_1,filter_2,16,brightness_map)

			Bg.append(dist)

		Bg=np.array(Bg)

		# print(Bg.shape)

		Bg=np.mean(Bg,axis=0)

		fig=plt.figure()
		plt.imshow(min_max_scaling(Bg),cmap='gray')

		plt.savefig(f'./{photo_index+1}/Bg_{photo_index+1}.png')


		"""
		Generate Color Map
		Perform color binning or clustering
		"""
		height,width,channels=image_colour.shape
		kmeans = KMeans(n_clusters = 16, random_state = 1)
		input=np.reshape(image_colour,((height*width),channels))
		labels=kmeans.fit_predict(input)
		# texton_map=kmeans.labels_

		color_map=np.reshape(labels,(height,width))
		


		"""
		Generate Color Gradient (Cg)
		Perform Chi-square calculation on Color Map
		Display Cg and save image as Cg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""

		Cg=[]
		for i in range(len(half_disc_bank_flattened)-1):

			filter_1=half_disc_bank_flattened[i]
			filter_2=half_disc_bank_flattened[i+1]

			dist=chi_square_distance(filter_1,filter_2,16,color_map)

			Cg.append(dist)

		Cg=np.array(Cg)

		Cg=np.mean(Cg,axis=0)

		fig=plt.figure()
		plt.imshow(min_max_scaling(Cg),cmap='gray')

		plt.savefig(f'./{photo_index+1}/Cg_{photo_index+1}.png')






		"""
		Read Sobel Baseline
		use command "cv2.imread(...)"
		"""

		sobel_baseline=cv2.imread(f'../BSDS500/SobelBaseline/{photo_index+1}.png',0)

		"""
		Read Canny Baseline
		use command "cv2.imread(...)"
		"""

		canny_baseline=cv2.imread(f'../BSDS500/CannyBaseline/{photo_index+1}.png',0)


		"""
		Combine responses to get pb-lite output
		Display PbLite and save image as PbLite_ImageName.png
		use command "cv2.imwrite(...)"
		"""

		first_term=(Tg+Cg+Bg)/3
		second_term=(0.5*canny_baseline+0.5*sobel_baseline)

		final_image=np.multiply(first_term,second_term)

		fig=plt.figure()
		plt.imshow(final_image,cmap='gray')

		plt.savefig(f'./{photo_index+1}/PbLite_{photo_index+1}.jpg')

    
if __name__ == '__main__':
    main()
 


