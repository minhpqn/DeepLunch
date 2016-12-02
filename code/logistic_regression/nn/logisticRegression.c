#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define IN 256  // dimension of the input variables
#define OUT 256 // dimension of the output variables

/*
   Logistic regression to predict a character given the previous one
   */

int main()
{
	printf("--------Logistic regression---------\n");
	int a,b,c,i,x,y,div=0,cor,count;
	double denom, loss_fun, 
	       alpha=0.05, // starting learning rate
	       lastent=-100000000;

	FILE *f;
	char train_filename[]         = "train_sequence"; // name of the training set
	char validation_filename[]    = "validation_sequence"; //name of the validation set

	int curr, // current character in the text
	    next; // next character in the text

	/******* Variables of the Model *******/

	double W[IN*OUT], // weight matrix of the model (to be learn)
	       in[IN],  // input of the model
	       out[OUT], // output of the model, i.e., the prediction
	       err[OUT]; // error made by the model on the prediction


	// initialization of the weight matrix W:
	for (a=0; a<IN*OUT; a++) W[a]=0;


	/******* Optimization *******/

	int max_epochs = 100; // max number of epochs through file
	for (i=0; i < max_epochs; i++) {

		printf("epoch %i:\n", i+1);

		// open train file
		f       = fopen(train_filename, "rb");

		next     = 0;

		cor      = 0;
		loss_fun = 0;// loss function
		count    = 0;// count the number of characters

		//loop over the characters
		while(1){

			curr = next;
			fscanf(f, "%c", &next);
			count++;

			if(feof(f)) 
				break;

			// one-hot representation:
			// set input vector as in[n] = 1 if n == current letter, 
			// otherwise in[n] = 0
			for (b = 0; b < IN; b++) in[b] = 0;
			in[curr] = 1;

			for (b = 0; b < OUT; b++) out[b] = 0;//erase $out

			/*** forward propagation ***/
			// aka compute the prediction given the input
			// (in latex notation) out_k = \exp ( \sum_d W_{kd} * in_d ) / \sum_l ( \exp ( W_{ld} * in_d ) ) 

			// linear mapping:
			for (x = 0; x < OUT; x++) 
				for (y = 0; y < IN; y++) 
					out[x] += in[y] * W[x * IN + y]; // out = W * in

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


			loss_fun += log10(out[next]) / log10(2);

			// compute the value of the cost function up to $count characters
			printf("-- value of loss function on train: %lf %c", loss_fun / count, 13);


			/*** backpropagation ***/
			// aka compute the gradient for gradient descent

			// compute the error made by our prediction:
			// error = true value - prediction
			for (x = 0; x < OUT; x++) err[x] = 0 - out[x];
			err[next] = 1 - out[next];

			// update the weight matrix with the backpropagated error 
			// aka do a gradient descent update with a step of alpha:
			for (x=0; x<OUT; x++) 
				W[x*IN+curr] += alpha * err[x];

			// compute classication error:
			for (x = 0; x < OUT; x++) 
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

		// same as forward propagation for the training:
		while(1){

			curr = next;
			fscanf(f, "%c", &next);
			count++;

			if(feof(f)) 
				break;

			// one-hot representation:
			// set input vector as in[n] = 1 if n == current letter, 
			// otherwise in[n] = 0
			for (b = 0; b < IN; b++) in[b] = 0;
			in[curr] = 1;

			for (b = 0; b < OUT; b++) out[b] = 0;//erase $out

			// Compute the prediction:
			// 
			// (in latex notation) out_k = \exp ( \sum_d W_{kd} * in_d ) / \sum_l ( \exp ( W_{ld} * in_d ) ) 

			for (x = 0; x < OUT; x++) 
				for (y = 0; y < IN; y++) 
					out[x] += in[y] * W[x * IN + y]; // out = W * in

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


			loss_fun += log10(out[next]) / log10(2);

			// compute the value of the cost function up to $count characters
			printf("-- value of loss function on validation: %lf %c", loss_fun / count, 13);

			// compute classication error:
			for (x = 0; x < OUT; x++) 
				if (out[x] >= out[next]) 
					if (x != next) {cor++; break;}
		}
		printf("\n");

		printf("--> classification error on validation: %lf\n", ((float)cor) / count);

		if (lastent>loss_fun) div=1;

		//if (div) alpha*=0.5;

		lastent=loss_fun;

		fclose(f);
	}

	f = fopen("W_logistic", "wb");
	for (x=0; x<OUT; x++) for (y=0; y<IN; y++) fprintf(f, "%lf ", W[x*IN+y]);
	fclose(f);


	return 0;
}
