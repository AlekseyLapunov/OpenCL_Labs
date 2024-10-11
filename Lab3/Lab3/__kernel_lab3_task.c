#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

__kernel void solve(__global const int* N,
	__global const int* K,
	__global const int* numbersCount,
	__global char* numbers) {

	int id = get_global_id(0);
	int workers = get_global_size(0);
	const int num = *numbersCount;

	int part = num / workers;
	int last;
	if (num % workers != 0) {
		part++;
		last = num - (workers - 1) * part;
	}
	else last = part;

	int begin = id * part;
	int end = begin + part;

	if (id == (workers - 1))
		end = begin + last;

	__local int oddNumbers;
	barrier(CLK_LOCAL_MEM_FENCE);

	char kChar = (char)(*K) + '0';
	for (int i = begin; i < end; i++) {
		int counter = 0;
		for (int j = i * 12; j < 12 * (i + 1); j++) {
			if (numbers[j] == kChar)
				counter++;
		}

		if (counter != (*N)) {
			numbers[i * 12] = '#';
			continue;
		}

		int sum = 0;
		for (int j = i * 12 + 1; j < 12 * (i + 1); j++) {
			sum += (numbers[j] - '0');
		}

		if (sum % 2 != 0) {
			atomic_inc(&oddNumbers);
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	if (id == 0)
		printf("Kernel Function %s | WORKER %d: Odd numbers = %d\n", __FUNCTION__, id, oddNumbers);
}
