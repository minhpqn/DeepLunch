#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define IN 256 // dimension of the input variables
#define HIDDEN 100 // number of neurons in the hidden layer 
#define OUT 256 // dimension of the output variables

int main()
{
	printf("--------Recurrent neural Network (1 hidden layer)---------\n");
	int a,b,c,i,x,y,div=0,cor,count;
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
	       in[IN],//input of the model
	       out[OUT],//output of the model, i.e., the prediction
	       hidden[HIDDEN], // hidden layer
	       oldhidden[HIDDEN],
	       errO[OUT], // error made by the model on the prediction
	       errH[HIDDEN]; // error made by the model on the hidden layer

	// random initialization of the weight matrices
	// to obtain different neuron activations  
	// This is a standard trick used to train neural network
	srand(1);
	for (a=0; a < IN*HIDDEN; a++) 
		W1[a] = ((rand()/(float)RAND_MAX)-0.5)/2;
	for (a=0; a < HIDDEN*OUT; a++) 
		W2[a] = ((rand()/(float)RAND_MAX)-0.5)/2;
	for (a=0; a < HIDDEN*HIDDEN; a++) 
		W3[a] = ((rand()/(float)RAND_MAX)-0.5)/2;

	for (a=0; a < HIDDEN; a++) 		
		hidden[a]= 0;


	/******* Optimization *******/

	int max_epochs = 100; // max number of epochs through file
	for (i = 0; i < max_epochs; i++) {

		printf("epoch %i:\n", i+1);

		// open train file
		f       = fopen(train_filename, "rb");

		cor      = 0;
		next     = 0;

		loss_fun = 0;// loss function
		count    = 0;// count the number of characters

		for (a=0; a < HIDDEN; a++) 		
			hidden[a]= 0;

		//loop over the characters
		while(1){

			curr = next;
			fscanf(f, "%c", &next);
			count++;

			if(feof(f)) 
				break;

			// copy $hidden in $oldhidden
			for(x=0; x< HIDDEN; x++)
				oldhidden[x]= hidden[x];

			// one-hot representation:
			// set input vector as in[n] = 1 if n == current letter, 
			// otherwise in[n] = 0
			for (b=0; b<IN; b++) 
				in[b] = 0;
			in[curr] = 1;

			for (b = 0; b < HIDDEN; b++) 
				hidden[b] = 0;	//erase $hidden
			for (b = 0; b < OUT; b++) 
				out[b] = 0;	//erase $out


			/*** forward propagation ***/
			// aka compute prediction given input

			// 1) propagation to hidden:

			// linear mapping from input:
			for (x=0; x<HIDDEN; x++)
				hidden[x]+=W1[x*IN+curr];
			// linear mapping from the previous hidden variables:
			for (x=0; x<HIDDEN; x++)
				for(y=0;y<HIDDEN; y++)
					hidden[x] += W3[x*HIDDEN+y] * oldhidden[y] ;


			//activation of hidden with a sigmoid 
			for (x = 0; x < HIDDEN; x++) 
			{
				// threshold $hidden to avoid numerical issues:
				if (hidden[x] > 50) hidden[x] = 50;
				if (hidden[x] < -50) hidden[x] = -50;

				hidden[x] = 1/(1 + exp(-hidden[x]));// sigmoid function
			}

			// 2) propagation to out:

			// linear mapping:
			for (x=0; x<OUT; x++) 
				for (y=0; y<HIDDEN; y++) 
					out[x] += hidden[y] * W2[x * HIDDEN + y];

			//softmax:
			denom = 0;
			for (x = 0; x < OUT; x++) {
				// threshold prediction to avoid numerical issues:
				if (out[x] > 50)  out[x] = 50;
				if (out[x] < -50) out[x] = -50;

				out[x] = exp(out[x]);

				denom+=out[x];
			}
			// normalized the prediction
			for (x=0; x<OUT; x++) 
				out[x] /= denom; 

			loss_fun += log10(out[next])/ log10(2);

			// compute the value of the cost function up to $count characters
			printf("-- value of loss function on train: %lf %c", loss_fun / count, 13);


			/*** backpropagation ***/
			// aka compute the gradient for gradient descent


			// compute the error made by our prediction:
			// error = true value - prediction
			for (x=0; x<OUT; x++) 
				errO[x] = 0 - out[x];
			errO[next] = 1 - out[next];

			// computing the error on the hidden layer regarding to the output
			for (x=0; x<HIDDEN; x++) errH[x]=0;
			for (x=0; x<OUT; x++) for (y=0; y<HIDDEN; y++) 
				errH[y] += errO[x]*W2[x*HIDDEN+y];
			for (y=0; y<HIDDEN; y++) 
				errH[y] = errH[y]*hidden[y]*(1-hidden[y]);


			// update the weight matrices with the backpropagated error 
			// aka do a gradient descent update with a step of alpha:

			// update W1
			for (x=0; x<HIDDEN; x++) 
				W1[x*IN+curr] += alpha*errH[x];
			//update of W2
			for (x=0; x<OUT; x++) 
				for (y=0; y<HIDDEN; y++) 
					W2[x*HIDDEN+y] += alpha*errO[x]*hidden[y];

			// update W3
			for (x=0; x<HIDDEN; x++) 
				for(y=0;y<HIDDEN; y++)
					W3 [x * HIDDEN+y] += alpha*errH[x] * oldhidden[y];

			// compute classication error:
			for (x = 0; x < OUT; x++) 
				if (out[x] >= out[next]) 
					if (x != next) {cor++; break;}

		}
		printf("\n");
		printf("--> classification error on train: %lf\n", ((float)cor) / count );
		fclose(f);

		//if(i==3) return;
		
		// computing error on the validation set:
		f = fopen(validation_filename, "rb");

		loss_fun = 0;
		cor      = 0;
		next     = 0;
		count    = 0;

		for (a=0; a < HIDDEN; a++) 		
			hidden[a]= 0;

		// same as forward propagation for the training:
		while(1){

			// copy $hidden in $oldhidden
			for(x=0; x< HIDDEN; x++)
				oldhidden[x]= hidden[x];

			curr = next;
			fscanf(f, "%c", &next);
			count++;

			if(feof(f)) 
				break;

			// one-hot representation:
			// set input vector as in[n] = 1 if n == current letter, 
			// otherwise in[n] = 0
			for (b=0; b<IN; b++) 
				in[b] = 0;
			in[curr] = 1;

			for (b = 0; b < HIDDEN; b++) 
				hidden[b] = 0;	//erase $hidden
			for (b = 0; b < OUT; b++) 
				out[b] = 0;	//erase $out


			/*** forward propagation ***/
			// aka compute prediction given input

			// 1) propagation to hidden:

			// linear mapping from input:
			for (x=0; x<HIDDEN; x++)
				hidden[x]+=W1[x*IN+curr];
			// linear mapping from the previous hidden variables:
			for (x=0; x<HIDDEN; x++)
				for(y=0;y<HIDDEN; y++)
					hidden[x] += W3[x*HIDDEN+y] * oldhidden[y] ;


			//activation of hidden with a sigmoid 
			for (x = 0; x < HIDDEN; x++) 
			{
				// threshold $hidden to avoid numerical issues:
				if (hidden[x] > 50) hidden[x] = 50;
				if (hidden[x] < -50) hidden[x] = -50;

				hidden[x] = 1/(1 + exp(-hidden[x]));// sigmoid function
			}

			// 2) propagation to out:

			// linear mapping:
			for (x=0; x<OUT; x++) 
				for (y=0; y<HIDDEN; y++) 
					out[x] += hidden[y] * W2[x * HIDDEN + y];

			//softmax:
			denom = 0;
			for (x = 0; x < OUT; x++) {
				// threshold prediction to avoid numerical issues:
				if (out[x] > 50)  out[x] = 50;
				if (out[x] < -50) out[x] = -50;

				out[x] = exp(out[x]);

				denom+=out[x];
			}
			// normalized the prediction
			for (x=0; x<OUT; x++) 
				out[x] /= denom; 

			loss_fun += log10(out[next])/ log10(2);

			// compute the value of the cost function up to $count characters
			printf("-- value of loss function on validation: %lf %c", loss_fun / count, 13);


			// compute classication error:
			for (x = 0; x < OUT; x++) 
				if (out[x] >= out[next]) 
					if (x != next) {cor++; break;}

		}

		printf("\n");
		printf("--> classification error on validation: %lf\n", ((float)cor) / count);
		//stochastic gradient: learning rate => constant and then descreasing by constant factor

		// if no significant improvment reduce learning rate for the next epoch
		if (lastent > loss_fun*1.001) 
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

