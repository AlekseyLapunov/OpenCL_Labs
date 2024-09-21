__kernel void SimpleWithArray (__global int *n, __global const int* size) {
	for (int i = 0; i < (*size); i++) {
		n[i] = n[i]*2;
	}
}