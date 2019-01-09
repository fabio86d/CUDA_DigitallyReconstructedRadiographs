/**
* Fabio D'Isidoro, ETH Zurich, May 2017
*
* Implementation of a CUDA-based Cpp library for fast DRR generation with GPU acceleration
*
* Based both on the description found in the “Improved Algorithm” section in Jacob’s paper (1998)
* https://www.researchgate.net/publication/2344985_A_Fast_Algorithm_to_Calculate_the_Exact_Radiological_Path_Through_a_Pixel_Or_Voxel_Space
* and on the implementation suggested in Greef et al 2009
* https://www.ncbi.nlm.nih.gov/pubmed/19810482
*
*
* Header for Class Siddon. 
*
* The class loads a CT scan onto the GPU memory. The function generateDRR can be called multiple times 
* to generate Digitally Reconstructed Ragiographs.
*
* - The class constructor needs the following variables:
*
* @param NumThreadsPerBlock: Number of threads per block to launch kernel (number of blocks will be determined based on DRRsize)
* @param movImgArray: 1D array of CT values
* @param MovSize: array relative to the size of the Moving Image (CT scan) (needed to properly index the 1D array obtained from the Moving Image)
* @param MovSpacing: array relative to the spacing of the Moving Image (CT scan)
* @param X0, Y0, Z0: physical coordinates of side planes
* @param DRRsize: size array of the output DRR image (needed to properly index the 1D array obtained from the output DRR image)
*
* - The function generate DRR must be called with the following variables:
*
* @param source: array of (transformed) source physical coordinates
* @param DestArray: C-ordered 1D array of physical coordinates relative to the (transformed) output DRR image.
* @param drrArray: output, 1D array for output values of projected CT densities
*
* Data type for image values is float (itk.F)
* Data type for physical coordinates (usually itk.D) is float.
*
*/


class SiddonGpu {

public:

	static const int Siddon_Dimension = 3;

	SiddonGpu(); // default constructor
	SiddonGpu(int *NumThreadsPerBlock,
			  float *movImgArray,
			  int *MovSize,
			  float *MovSpacing,
			  float X0, float Y0, float Z0,
			  int *DRRSize); // overloaded constructor
	~SiddonGpu(); // destructor

	// function to generate DRR
	void generateDRR(float *source,
					 float *DestArray,
					 float *drrArray);

private:

	// --- declaration of public members (d_ prefix means "for device memory", m_ prefix class members) --- 
	// to launch kernel
	int m_NumThreadsPerBlock[Siddon_Dimension];

	// moving image
	int m_MovSize[Siddon_Dimension];
	int m_movImgMemSize;
	float m_X0, m_Y0, m_Z0;
	float *m_d_movImgArray; // device copy
	int *m_d_MovSize; // device copy
	float *m_d_MovSpacing; // device copy

	// DRR
	int m_DRRsize[Siddon_Dimension];
	int m_DRRsize0;
	int m_DrrMemSize;
	int m_DestMemSize;

};
