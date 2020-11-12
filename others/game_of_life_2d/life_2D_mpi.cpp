#include <mpi.h>
#include <iostream>
#include <string>
#include <fstream>
#include <stdlib.h>

using namespace std;

#define NUM_ELEMENTS 100
#define GENERATIONS 10
#define OUTPOINTS 10

int main(int argc, char* argv[])
{
	int size, rank, tag, rc, N, generations, outPoints, s;
	MPI_Status Stat;
	ofstream output("output.txt"); 	// output file

	N = NUM_ELEMENTS;
	generations = GENERATIONS;
	outPoints = OUTPOINTS;

    // init MPI
	rc = MPI_Init(&argc,&argv);
	if (rc != 0) {cout << "Error starting MPI." << endl; MPI_Abort(MPI_COMM_WORLD, rc);}
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0){     		// Rank 0
		srand(time(NULL));
		s = N / size;							// how many slices 
		int Board[N][N];	
		for (int i = 0; i < N; i++){			// read file into array
			for (int j = 0; j < N; j++)
                Board[i][j] = rand() % 2;
		}

		// chech the board
		// cout << "Grid-" << N << "x" << N << ", NumSteps-" << GENERATIONS << ", OutPoints-" << OUTPOINTS << endl;
		// for (int i = 0; i < N; i++)
		// {
		// 	for (int j = 0; j < N; j++)
		// 		cout << Board[i][j] << " ";
		// 	cout << endl;
		// }

		// SENDING INITIAL INFORMATION (N, k, #generations, output points) TO EVERYONE
		int info[4];
		info[0] = N;
        info[1] = s;    		// num of slices 
        info[2] = generations;  // sounds like num of steps
        info[3] = outPoints;
		for (int dest = 0; dest < size; dest++)
		{
			cout << "R" << rank << ": sending input-info to " << dest << "..." << endl;
			MPI_Send(&info, 4, MPI_INT, dest, 1, MPI_COMM_WORLD); 	// send info	| TODO with MPI_Collec...
		}

		int slice[N/size][N];	
		for (int z = 0; z < size; z++)
		{
			for (int k = 0; k < s; k++)								// num. slides that a rank holds
				for (int l = 0; l < N; l++) 
					slice[k][l] = Board[k + (z*s)][l];	       		// cut a slice from the the board

			cout << "R" << rank << ": sending slice-info (slide[" << (0+z*s) << "-" << (s-1+z*s) << "]) to " << z << "..." << endl;
			MPI_Send(&slice, N*s, MPI_INT, z, 1, MPI_COMM_WORLD);	// and send it	| TODO with MPI_Collec...
		}
	}

	// RECEIVED INITIAL INFORMATION
	int localinfo[4];		    // local info for initial information
	MPI_Recv(&localinfo, 4, MPI_INT, 0, 1, MPI_COMM_WORLD, &Stat);	// receive info	| TODO with MPI_Collec...

	// assign variables on each rank
	N = localinfo[0];				//
	s = localinfo[1];				//
	generations = localinfo[2];		//
	outPoints = localinfo[3];		//

	int myslice[s][N];				// my own slice of the board
	MPI_Recv(&myslice, localinfo[0]*localinfo[1], MPI_INT, 0, 1, MPI_COMM_WORLD, &Stat);
	
	int todown[N];	int toup[N]; int fromdown[N]; int fromup[N]; 	// arrays to send and to receive
	for (int g = 1; g <= generations; g++) 							// generations forloop
	{	
		if (rank != size-1)			// all except for last send down
		{
			for (int j = 0; j < N; j++)
                todown[j] = myslice[s-1][j];
			MPI_Send(&todown, N, MPI_INT, rank+1, 1, MPI_COMM_WORLD);

		} else {
			for (int k = 0; k < N; k++)
                fromdown[k] = 0;	// last one generates empty stripe "from down"
        }

		if (rank != 0)				// all except for first receive from up
		{
			MPI_Recv(&fromup, N, MPI_INT, rank-1, 1, MPI_COMM_WORLD, &Stat);	

		} else {
            for (int k = 0; k < N; k++)
            	fromup[k] = 0;		// first one generats empty line "from up"
        }
	
		if (rank != 0)				// all except for first send up
		{
			for (int j = 0; j < N; j++)
                toup[j] = myslice[0][j];

			MPI_Send(&toup, N, MPI_INT, rank-1, 1, MPI_COMM_WORLD);
		}
	
		if (rank != size-1)			// all except for last receive from down
		{
			MPI_Recv(&fromdown, N, MPI_INT, rank+1, 1, MPI_COMM_WORLD, &Stat);
		}

		// COUNTING ON NEIGHBORS
		int sum = 0; 					// sum of neighbours
		int mynewslice[s][N];
		for (int x = 0; x < s; x++) 	// for each row
		{	
			for (int y = 0; y < N; y++) // for each column
			{
				if (x == 0 && y == 0) 			// upper-left cell
					sum = myslice[x+1][y] + myslice[x+1][y+1] + myslice[0][y+1] + fromup[0] + fromup[1];
				else if (x == 0 && y == N-1) 	// upper-right cell
					sum = myslice[x][y-1] + myslice[x+1][y-1] + myslice[x+1][y] + fromup[N-1] + fromup[N-2];
				else if (x == s-1 && y == 0) 	// lower-left cell
					sum = myslice[x][y+1] + myslice[x-1][y+1] + myslice[x-1][y] + fromdown[0] + fromdown[1];
				else if (x == s-1 && y == N-1) 	// lower-right cell
					sum = myslice[x-1][y] + myslice[x-1][y-1] + myslice[x][y-1] + fromdown[N-1] + fromdown[N-2];
				else // not corner cells    
				{
					if (y == 0) 			// leftmost line, not corner
						sum = myslice[x-1][y] + myslice[x-1][y+1] + myslice[x][y+1] + myslice[x+1][y+1] + myslice[x+1][y];
					else if (y == N-1) 		// rightmost line, not corner
						sum = myslice[x-1][y] + myslice[x-1][y-1] + myslice[x][y-1] + myslice[x+1][y-1] + myslice[x+1][y];
					else if (x == 0) 		// uppermost line, not corner
						sum = myslice[x][y-1] + myslice[x+1][y-1] + myslice[x+1][y] + myslice[x+1][y+1] + myslice[x][y+1] + fromup[y-1] + fromup[y] + fromup[y+1];
					else if (x == s-1) 		// lowermost line, not corner
						sum = myslice[x-1][y-1] + myslice[x-1][y] + myslice[x-1][y+1] + myslice[x][y+1] + myslice[x][y-1] + fromdown[y-1] + fromdown[y] + fromdown[y+1];
					else 					// general case, any cell within
						sum = myslice[x-1][y-1] + myslice[x-1][y] + myslice[x-1][y+1] + myslice[x][y+1] + myslice[x+1][y+1] + myslice[x+1][y] + myslice[x+1][y-1] + myslice[x][y-1];
				}
				
				// PUT THE NEW VALUE OF A CELL
				if (myslice[x][y] == 1 && (sum == 2 || sum == 3))
					mynewslice[x][y] = 1;
				else if (myslice[x][y] == 1 && sum > 3)
					mynewslice[x][y] = 0;
				else if (myslice[x][y] == 1 && sum < 1)
					mynewslice[x][y] = 0;
				else if (myslice[x][y] == 0 && sum == 3)
					mynewslice[x][y] = 1;
		 		else
				 	mynewslice[x][y] = 0;
			
			}
		}
	
		// copy new slice onto myslice
		for (int x = 0; x < s; x++)
			for (int y = 0; y < N; y++)
				myslice[x][y] = mynewslice[x][y];

		// PRINTING THE RESULT TO FILE
		if (g / outPoints < 1)				// s-th generation, send everything to node 0
		{
			if (rank == 0)
			{
				int aBoard[s][N];
				output << "Generation " << g << ":" << endl;
				for (int x = 0; x < s; x++) // put your own slice
				{
					for (int y = 0; y < N; y++)
						output << myslice[x][y];
					output << endl;
				}
				for (int i = 1; i < size; i++)
				{
					MPI_Recv(&aBoard, N*s, MPI_INT, i, 1, MPI_COMM_WORLD, &Stat); // receive all others
					for (int x = 0; x < s; x++)
					{
						for (int y = 0; y < N; y++)
							output << aBoard[x][y];
						output << endl;
					}
				}
				output << endl << endl;
			}
			else
				MPI_Send(&myslice, N*s, MPI_INT, 0,1, MPI_COMM_WORLD);
		}	
	} // end of generation loop

    output.close();

    MPI_Finalize();

    return 0;
}