/**
 * @file    parallel_heat_solver.h
 * @author  xlogin00 <xlogin00@stud.fit.vutbr.cz>
 *
 * @brief   Course: PPP 2021/2022 - Project 1
 *
 * @date    2022-MM-DD
 */

#ifndef PARALLEL_HEAT_SOLVER_H
#define PARALLEL_HEAT_SOLVER_H

#include "base_heat_solver.h"

using namespace std;

/**
 * @brief The ParallelHeatSolver class implements parallel MPI based heat
 *        equation solver in 2D using 1D and 2D block grid decomposition.
 */
class ParallelHeatSolver : public BaseHeatSolver
{
    //============================================================================//
    //                            *** BEGIN: NOTE ***
    //
    // Modify this class declaration as needed.
    // This class needs to provide at least:
    // - Constructor which passes SimulationProperties and MaterialProperties
    //   to the base class. (see below)
    // - Implementation of RunSolver method. (see below)
    // 
    // It is strongly encouraged to define methods and member variables to improve 
    // readability of your code!
    //
    //                             *** END: NOTE ***
    //============================================================================//
    
public:
    /**
     * @brief Constructor - Initializes the solver. This should include things like:
     *        - Construct 1D or 2D grid of tiles.
     *        - Create MPI datatypes used in the simulation.
     *        - Open SEQUENTIAL or PARALLEL HDF5 file.
     *        - Allocate data for local tile.
     *        - Initialize persistent communications?
     *
     * @param simulationProps Parameters of simulation - passed into base class.
     * @param materialProps   Parameters of material - passed into base class.
     */
    ParallelHeatSolver(SimulationProperties &simulationProps, MaterialProperties &materialProps);
    virtual ~ParallelHeatSolver();

    /**
     * @brief Run main simulation loop.
     * @param outResult Output array which is to be filled with computed temperature values.
     *                  The vector is pre-allocated and its size is given by dimensions
     *                  of the input file (edgeSize*edgeSize).
     *                  NOTE: The vector is allocated (and should be used) *ONLY*
     *                        by master process (rank 0 in MPI_COMM_WORLD)!
     */
    virtual void RunSolver(std::vector<float, AlignedAllocator<float> > &outResult);

protected:
    MPI_Comm MPI_COL_COMM;
    
    MPI_Datatype MPI_TWO_ROW_INT;
    MPI_Datatype MPI_TWO_COL_INT;
    MPI_Datatype MPI_TWO_ROW_FLOAT;
    MPI_Datatype MPI_TWO_COL_FLOAT;
    MPI_Datatype MPI_SUBMAT_FLOAT;
    MPI_Datatype MPI_SUBMAT_RECV_FLOAT;

    MPI_Win winNewTemp;
    MPI_Win winTemp;
    AutoHandle<hid_t> m_fileHandle;                             ///< Output HDF5 file handle.
    
    int *domainMap;
    float *domainParams;
    float *temp;
    float *newTemp;

    float *res;
    int *displs;
    int *sendcounts;
    int m_rank;     ///< Process rank in global (MPI_COMM_WORLD) communicator.
    int m_size;     ///< Total number of processes in MPI_COMM_WORLD.
    
    int p_cols;
    int p_rows;

    int edgeSize;
    int rowSize;
    int colSize;
    int extRowSize;
    int extColSize;
    
    int middleCol;

    int topRank;
    int botRank;
    int leftRank;
    int rightRank;

    int leftPadding;
    int rightPadding;

    int topSendPadding;
    int botSendPadding;
    int leftSendPadding;
    int rightSendPadding;
    
    int topRecvPadding;
    int botRecvPadding;
    int leftRecvPadding;
    int rightRecvPadding;

    int diskWriteIntensity;

    bool modeRMA;
    bool isTop;
    bool isBottom;
    bool isLeft;
    bool isRight;
    bool isMiddle;
    bool write = false;

    void parallelStoreDataIntoFile(int iteration);
};

#endif // PARALLEL_HEAT_SOLVER_H
