__kernel void solve(__global const int* N,
					__global const int* K,
					__global const int* numbersCount,
					__global char* numbers) {
						
	int id		  = get_global_id(0);
	int workers   = get_global_size(0);
	const int num = *numbersCount;

	int part = num/workers;
	int last;
	if (num % workers != 0) {
		part++;
		last = num - (workers - 1)*part;
	}
	else last = part;
	
	int begin = id*part;
	int end	  = begin + part;

	if (id == (workers - 1))
		end = begin + last;

	char kChar = (char)(*K) + '0';
	for (int i = begin; i < end; i++) {
		int counter = 0;
		for (int j = i*12; j < 12*(i + 1); j++) {
			if (numbers[j] == kChar)
				counter++;
		}
		if (counter != (*N))
			numbers[i*12] = '#';
	}
}
