#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define IN 256 // dimension of the input variables
#define HIDDEN 100 // number of neurons in the hidden layer 
#define OUT 256 // dimension of the output variables
#define BPTT 6 // truncation for the recurrence

int main()
{
	printf("--------Recurrent neural Network (1 hidden layer)---------\n");
	int c,div=0,cor,count;
	double d, loss_fun, denom,
	       alpha	= 0.03, // starting learning rate
	       lastent	= -100000000;

	FILE *f;
	char train_filename[]      = "train_sequence"; // name of the training set
	char validation_filename[] = "validation_sequence"; //name of the validation set

	int curr, // current character in the text
	    next; // next character in the text

	/******* Variables of the Model *******/


	double W1[IN*HIDDEN],//weight matrix between input and hidden layer (to be learn)
	       W2[HIDDEN*OUT],//weight matrix between output and hidden layer (to be learn)
	       W3[HIDDEN*HIDDEN],//weight matrix for the recurrence
	       dW1[IN*HIDDEN], // gradient for W1
	       dW3[HIDDEN*HIDDEN], // gradient for W3
	       in[BPTT][IN],//input of the model
	       out[OUT],//output of the model, i.e., the prediction
	       hidden[BPTT][HIDDEN], // hidden layer
	       errO[OUT], // error made by the model on the prediction
	       errHold[HIDDEN], // error made by the model on the hidden layer
	       errH[HIDDEN]; // error made by the model on the hidden layer

	// random initialization of the weight matrices
	// to obtain different neuron activations  
	// This is a standard trick used to train neural network
	srand(1);
	for (int a=0; a < IN*HIDDEN; a++) 
		W1[a] = ((rand()/(float)RAND_MAX)-0.5)/2;
	for (int a=0; a < HIDDEN*OUT; a++) 
		W2[a] = ((rand()/(float)RAND_MAX)-0.5)/2;
	for (int a=0; a < HIDDEN*HIDDEN; a++) 
		W3[a] = ((rand()/(float)RAND_MAX)-0.5)/2;



	/******* Optimization *******/

	int max_epochs = 100; // max number of epochs through file
	for (int i = 0; i < max_epochs; i++) {

		printf("epoch %i:\n", i+1);

		// open train file
		f       = fopen(train_filename, "rb");

		cor      = 0;
		next     = 0;
		loss_fun = 0;// loss function
		count    = 0;// count the number of characters

		for(int x = 0; x < BPTT; x++)
			for(int a = 0; a < HIDDEN; a++)
				hidden[x][a] = 0;
		for(int x = 0; x < BPTT; x++)
			for(int a = 0; a < IN; a++)
				in[x][a] = 0;

		//loop over the characters
		while(1){

			count++;
			curr = next;

			fscanf(f, "%c", &next);
			if(feof(f)) break;

			/**** Update variables ****/


			// update $hidden:
			for(int a = BPTT - 2; a >= 0 ;  a--)
				for(int x = 0; x < HIDDEN; x++)
					hidden[a + 1][x] = hidden[a][x];

			// update $in:
			for(int a = BPTT - 2; a >= 0; a--)
				for(int x = 0; x < IN; x++) 
					in[a + 1][x] = in[a][x];

			// one-hot representation:
			// set input vector as in[n] = 1 if n == current letter, 
			// otherwise in[n] = 0
			for(int x = 0; x < IN; x++) in[0][x] = 0;
			in[0][curr] = 1;

			for(int x = 0; x < HIDDEN; x++) hidden[0][x] = 0; //erase $hidden
			for(int x = 0; x < OUT; x++) 	out[x] = 0; //erase $out


			/***********************************************************/
			/**** forward propagation ****/
			// aka compute prediction given input

			// 1) propagation to hidden:

			// linear mapping from input:
			for(int x = 0; x < HIDDEN; x++)
				for(int y = 0; y < IN; y++)
					hidden[0][x] += W1[x * IN + y] * in[0][y];

			// linear mapping from the previous hidden variables:
			for(int x = 0; x < HIDDEN; x++) for(int y = 0; y < HIDDEN; y++)
				hidden[0][x] += W3[x * HIDDEN + y] * hidden[1][y] ;


			//activation of hidden with a sigmoid 
			for(int x = 0; x < HIDDEN; x++) 
			{
				// threshold $hidden to avoid numerical issues:
				if (hidden[0][x] > 50) hidden[0][x] = 50;
				if (hidden[0][x] < -50) hidden[0][x] = -50;

				hidden[0][x] = 1 / ( 1 + exp( - hidden[0][x] ) );// sigmoid function
			}

			// 2) propagation to out:

			// linear mapping:
			for(int x=0; x<OUT; x++) 
				for(int y=0; y<HIDDEN; y++) 
					out[x] += W2[x * HIDDEN + y] * hidden[0][y];

			//softmax:
			denom = 0;
			for(int x = 0; x < OUT; x++) {
				// threshold prediction to avoid numerical issues:
				if (out[x] > 50)  out[x] = 50;
				if (out[x] < -50) out[x] = -50;

				out[x] = exp(out[x]);

				denom += out[x];
			}
			// normalized the prediction
			for(int x = 0; x < OUT; x++) 
				out[x] /= denom; 

			loss_fun += log10(out[next]) / log10(2);

			// compute the value of the cost function up to $count characters
			printf("-- value of loss function on train: %lf %c", loss_fun / count, 13);


			/***********************************************************/
			/**** backpropagation ****/
			// aka compute the gradient for gradient descent


			// compute the error made by our prediction:
			// error = true value - prediction
			for(int x = 0; x < OUT; x++) 
				errO[x] = 0 - out[x];
			errO[next] = 1 - out[next];

			// computing the error on the hidden layer regarding to the output
			for(int x = 0; x < HIDDEN; x++) errH[x] = 0;

			for(int x = 0; x < OUT; x++) 
				for(int y = 0; y < HIDDEN; y++) 
					errH[y] += errO[x] * W2[x * HIDDEN + y];

			// compute gradient for W3 and W1:
			for(int x = 0; x < HIDDEN * HIDDEN; x++) dW3[x] = 0;
			for(int x = 0; x < IN * HIDDEN; x++) dW1[x] = 0;

			for(int  a = 0; a < BPTT-1; a++){
				// add sigmoid gradient to the error:
				for(int y = 0; y < HIDDEN; y++) 
					errH[y] *= hidden[a][y] * (1 - hidden[a][y]);

				// compute the gradient of W1 up to $a in time:
				for(int x = 0; x < HIDDEN; x++) 
					for(int y = 0; y < IN; y++)
						dW1[x * IN + y] += errH[x] * in[a][y];

				// compute the gradient of W3 up to $a in time:
				for(int x = 0; x < HIDDEN; x++) 
					for(int y = 0; y < HIDDEN; y++)
						dW3[x * HIDDEN + y] += errH[x] * hidden[a + 1][y];
				
				// propagate the error through time:
				for(int x = 0; x < HIDDEN; x++) 
					errHold[x] = 0;
				for(int x = 0; x < HIDDEN; x++) 
					for(int y = 0; y < HIDDEN; y++) 
						errHold[y] += errH[x] * W3[x * HIDDEN + y];
				
				for(int x = 0; x < HIDDEN; x++) 
					errH[x] = errHold[x];
				
			}
			

			// update the weight matrices with the backpropagated error 
			// aka do a gradient descent update with a step of alpha:

			//update of W2
			for(int x=0; x<OUT; x++) 
				for(int y=0; y<HIDDEN; y++) 
					W2[x*HIDDEN+y] += alpha * errO[x] * hidden[0][y];

			// update W1
			for(int x=0; x<HIDDEN; x++) 
				for(int y=0; y<IN; y++) 
					W1[x*IN+y] += alpha * dW1[x*IN+y];

			// update W3
			for(int x=0; x<HIDDEN; x++) 
				for(int y=0;y<HIDDEN; y++)
					W3 [x * HIDDEN + y] += alpha * dW3[x * HIDDEN + y];


			// compute classication error:
			for(int x = 0; x < OUT; x++) 
				if (out[x] >= out[next]) 
					if (x != next) {cor++; break;}

		}
		printf("\n");
		printf("--> classification error on train: %lf\n", ((float)cor) / count );
		fclose(f);

		// computing error on the validation set:
		f = fopen(validation_filename, "rb");

		loss_fun = 0;
		cor      = 0;
		next     = 0;
		count    = 0;

		for(int x = 0; x < BPTT; x++)
			for(int a = 0; a < HIDDEN; a++)
				hidden[x][a] = 0;
		for(int x = 0; x < BPTT; x++)
			for(int a = 0; a < IN; a++)
				in[x][a] = 0;

		// same as forward propagation for the training:
		while(1){

			curr = next;
			fscanf(f, "%c", &next);
			count++;

			if(feof(f)) 
				break;

			/**** Update variables ****/


			// update $hidden:
			for(int x = 0; x < BPTT - 1; x++)
				for(int a = 0; a < HIDDEN; a++)
					hidden[x+1][a] = hidden[x][a];

			// update $in:
			for(int x = 0; x < BPTT - 1; x++)
				for(int b=0; b<IN; b++) 
					in[x+1][b] = in[x][b];

			// one-hot representation:
			// set input vector as in[n] = 1 if n == current letter, 
			// otherwise in[n] = 0
			for(int b = 0; b < IN; b++) in[0][b] = 0;
			in[0][curr] = 1;

			for(int b = 0; b < HIDDEN; b++) hidden[0][b] = 0; //erase $hidden
			for(int b = 0; b < OUT; b++) 	out[b] = 0; //erase $out




			/*** forward propagation ***/
			// aka compute prediction given input

			// 1) propagation to hidden:

			// linear mapping from input:
			for(int x=0; x<HIDDEN; x++)
				hidden[0][x] += W1[x * IN + curr];
			// linear mapping from the previous hidden variables:
			for(int x=0; x<HIDDEN; x++)
				for(int y=0;y<HIDDEN; y++)
					hidden[0][x] += W3[x*HIDDEN+y] * hidden[1][y] ;


			//activation of hidden with a sigmoid 
			for(int x = 0; x < HIDDEN; x++) 
			{
				// threshold $hidden to avoid numerical issues:
				if (hidden[0][x] > 50) hidden[0][x] = 50;
				if (hidden[0][x] < -50) hidden[0][x] = -50;

				hidden[0][x] = 1/(1 + exp(-hidden[0][x]));// sigmoid function
			}

			// 2) propagation to out:

			// linear mapping:
			for(int x=0; x<OUT; x++) 
				for(int y=0; y<HIDDEN; y++) 
					out[x] += hidden[0][y] * W2[x * HIDDEN + y];

			//softmax:
			denom = 0;
			for(int x = 0; x < OUT; x++) {
				// threshold prediction to avoid numerical issues:
				if (out[x] > 50)  out[x] = 50;
				if (out[x] < -50) out[x] = -50;

				out[x] = exp(out[x]);

				denom+=out[x];
			}
			// normalized the prediction
			for(int x=0; x<OUT; x++) 
				out[x] /= denom; 

			loss_fun += log10(out[next])/ log10(2);

			// compute the value of the cost function up to $count characters
			printf("-- value of loss function on validation: %lf %c", loss_fun / count, 13);


			// compute classication error:
			for(int x = 0; x < OUT; x++) 
				if (out[x] >= out[next]) 
					if (x != next) {cor++; break;}

		}

		printf("\n");
		printf("--> classification error on validation: %lf\n", ((float)cor) / count);
		//stochastic gradient: learning rate => constant and then descreasing by constant factor

		// if no significant improvment reduce learning rate for the next epoch
		if (lastent > loss_fun*1.01) 
			div=1;

		if (div) 
			alpha*=0.5;

		if(alpha < 1e-8) break;

		printf("new learning rate: alpha = %f \n", alpha);

		// reset div = 0 (to control learning rate)
		div = 0;

		lastent=loss_fun;
		fclose(f);
	}

	return 0;
}


