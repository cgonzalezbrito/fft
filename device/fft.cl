#define windowLength 32

#define mask_left fft_index
#define mask_right l_addr
#define shift_pos N2
#define cosine x3.s0
#define sine x3.s1
#define wk x2

__kernel void fft (__global float2 *restrict input,
  __global float *restrict output, __local float2* l_data, uint N) {

  uint l_addr, i, j, k, fft_index,N2;
  uint4 br, index;
  float2 x1, x2, x3, x4, sum12, diff12, sum34, diff34;
  float absfft[windowLength], sum=0, fftAverage;

  for (j=0; j < N; j+=4){
    index = (uint4)(j,j+1,j+2,j+3);
    mask_left = N/2;
    mask_right = 1;
    shift_pos = 4;
    br = (index << shift_pos) & mask_left;
    br |= (index >> shift_pos) & mask_right;

    // Bit-reverse addresses
    while(shift_pos > 1){
      shift_pos -= 2;
      mask_left >>= 1;
      mask_right <<= 1;
      br |= (index << shift_pos) & mask_left;
      br |= (index >> shift_pos) & mask_right;
    }

    // Load global data
    x1 = input[br.s0];
    x2 = input[br.s1];
    x3 = input[br.s2];
    x4 = input[br.s3];

    sum12 = x1 + x2;
    diff12 = x1 - x2;
    sum34 = x3 + x4;
    diff34 = (float2)(x3.s1 - x4.s1, x4.s0 - x3.s0);
    l_data[j] = sum12 + sum34;
    l_data[j+1] = diff12 + diff34;
    l_data[j+2] = sum12 - sum34;
    l_data[j+3] = diff12 - diff34;
  }

  // Perform initial stages of the FFT - each of length N2*2
  for(N2 = 4; N2 < N; N2 <<= 1) {
    l_addr = 0;
    for(fft_index = 0; fft_index < N; fft_index += 2*N2) {
      x1 = l_data[l_addr];
      l_data[l_addr] += l_data[l_addr + N2];
      l_data[l_addr + N2] = x1 - l_data[l_addr + N2];
      for(j=1; j < N2; j++) {
        cosine = cos(M_PI_F*j/N2);
        sine = sin(M_PI_F*j/N2);
        wk = (float2)(l_data[l_addr+N2+j].s0*cosine + l_data[l_addr+N2+j].s1*sine,
                      l_data[l_addr+N2+j].s1*cosine - l_data[l_addr+N2+j].s0*sine);
        l_data[l_addr+N2+j] = l_data[l_addr+j] - wk;
        l_data[l_addr+j] += wk;
      }
      l_addr += 2*N2;
    }
  }

  for(j=0;j<N;j++){
    output[j]=l_data[j].s0;
    absfft[j]=sqrt(l_data[j].s0*l_data[j].s0 + l_data[j].s1*l_data[j].s1);
    sum += absfft[j];
  }

  fftAverage = sum/N;

  sum=0;

  for(int j=0;j<=N;j++){
    sum=sum+pow(absfft[j]-fftAverage,2);
  }

}
