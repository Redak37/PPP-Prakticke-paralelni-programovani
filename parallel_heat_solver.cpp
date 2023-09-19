/**
 * @file    parallel_heat_solver.cpp
 * @author  xlogin00 <xlogin00@stud.fit.vutbr.cz>
 *
 * @brief   Course: PPP 2020/2021 - Project 1
 *
 * @date    2021-MM-DD
 */

#include "parallel_heat_solver.h"
#include <iostream>
#include <iomanip>

using namespace std;

#define PADDING 2

//============================================================================//
//                            *** BEGIN: NOTE ***
//
// Implement methods of your ParallelHeatSolver class here.
// Freely modify any existing code in ***THIS FILE*** as only stubs are provided 
// to allow code to compile.
//
//                             *** END: NOTE ***
//============================================================================//

ParallelHeatSolver::ParallelHeatSolver(SimulationProperties &simulationProps,
                                       MaterialProperties &materialProps)
    : BaseHeatSolver(simulationProps, materialProps),
      m_fileHandle(H5I_INVALID_HID, static_cast<void (*)(hid_t )>(nullptr))
{
    MPI_Comm_size(MPI_COMM_WORLD, &m_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    m_simulationProperties.GetDecompGrid(p_cols, p_rows);
    MPI_Comm_split(MPI_COMM_WORLD, m_rank % p_cols, m_rank / p_cols, &MPI_COL_COMM);
   
    diskWriteIntensity = m_simulationProperties.GetDiskWriteIntensity();

    edgeSize = m_materialProperties.GetEdgeSize();
    const int extEdgeSize = edgeSize + 2 * PADDING;

    isTop = m_rank / p_cols == 0;
    isBottom = m_rank >= m_size - p_cols;
    isLeft = m_rank % p_cols == 0;
    isRight = m_rank % p_cols == p_cols - 1;
    isMiddle = (m_rank % p_cols == p_cols / 2);
    middleCol = (p_cols == 1 ? edgeSize / 2 : 0);

    leftPadding = isLeft ? 2 * PADDING : PADDING;
    rightPadding = isRight ? 2 * PADDING : PADDING;
   
    topRank = m_rank - p_cols;
    botRank = m_rank + p_cols;
    leftRank = m_rank - 1;
    rightRank = m_rank + 1;
    
    rowSize = edgeSize / p_cols;
    colSize = edgeSize / p_rows;
    extRowSize = rowSize + 2 * PADDING;
    extColSize = colSize + 2 * PADDING;
    const int subSize = extRowSize * extColSize;
    
    topSendPadding = PADDING + PADDING * extRowSize;
    botSendPadding = PADDING + (extColSize - 2 * PADDING) * extRowSize;
    leftSendPadding = PADDING + PADDING * extRowSize;
    rightSendPadding = (PADDING + 1) * extRowSize - 2 * PADDING;
    
    topRecvPadding = PADDING;
    botRecvPadding = (extColSize - PADDING) * extRowSize + PADDING;
    leftRecvPadding = PADDING * extRowSize;
    rightRecvPadding = (PADDING + 1) * extRowSize - PADDING;

    domainMap = new int[subSize]();
    domainParams = new float[subSize]();
    temp = new float[subSize]();
    newTemp = new float[subSize]();
    
    res = new float[(!m_rank) ? edgeSize * edgeSize : 0];
    displs = new int[(!m_rank) ? m_size : 0];
    sendcounts = new int[(!m_rank) ? m_size : 0];

    modeRMA = m_simulationProperties.IsRunParallelRMA();
    // Creating EMPTY HDF5 handle using RAII "AutoHandle" type
    //
    // AutoHandle<hid_t> myHandle(H5I_INVALID_HID, static_cast<void (*)(hid_t )>(nullptr))
    //
    // Thiscan be particularly useful when creating handle as class member!
    // Handle CANNOT be assigned using "=" or copy-constructor, yet it can be set
    // using ".Set(/* handle */, /* close/free function */)" as in:
    // myHandle.Set(H5Fopen(...), H5Fclose);
    
    // Requested domain decomposition can be queried by
    // m_simulationProperties.GetDecompGrid(/* tempS IN X */, /* tempS IN Y */)
    
    // 1. Open output file if its name was specified.
    if (!m_simulationProperties.GetOutputFileName().empty()) {
        write = true;
        if (m_simulationProperties.IsUseParallelIO()) {
            hid_t plist = H5Pcreate(H5P_FILE_ACCESS);
            H5Pset_fapl_mpio(plist, MPI_COMM_WORLD, MPI_INFO_NULL);

            m_fileHandle.Set(H5Fcreate(simulationProps.GetOutputFileName("par").c_str(),
                                       H5F_ACC_TRUNC, H5P_DEFAULT, plist), H5Fclose);

            H5Pclose(plist);
        } else if (!m_rank) {
            m_fileHandle.Set(H5Fcreate(simulationProps.GetOutputFileName("par").c_str(),
                                       H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), H5Fclose);
        }
    }

    MPI_Datatype MPI_MAT_INT;
    MPI_Datatype MPI_MAT_FLOAT;
    MPI_Datatype MPI_BLOCK_INT;
    MPI_Datatype MPI_BLOCK_FLOAT;

    vector<int> domMapSend;
    vector<float> domParamsSend;
    vector<float> tempSend;
    if (!m_rank) {
        fill_n(sendcounts, m_size, 1);

        for (int i = 0; i < edgeSize; i++) {
            domMapSend.resize((2 + i) * extEdgeSize + PADDING);
            domParamsSend.resize((2 + i) * extEdgeSize + PADDING);
            tempSend.resize((2 + i) * extEdgeSize + PADDING);
            domMapSend.insert(domMapSend.end(),
                           m_materialProperties.GetDomainMap().begin() + i * edgeSize,
                           m_materialProperties.GetDomainMap().begin() + (i + 1) * edgeSize);
            domParamsSend.insert(domParamsSend.end(),
                           m_materialProperties.GetDomainParams().begin() + i * edgeSize,
                           m_materialProperties.GetDomainParams().begin() + (i + 1) * edgeSize);
            tempSend.insert(tempSend.end(),
                           m_materialProperties.GetInitTemp().begin() + i * edgeSize,
                           m_materialProperties.GetInitTemp().begin() + (i + 1) * edgeSize);
        }
        
        const int extEdgeSize2 = extEdgeSize * extEdgeSize;
        domMapSend.resize(extEdgeSize2);
        domParamsSend.resize(extEdgeSize2);
        tempSend.resize(extEdgeSize2);

        MPI_Type_vector(extColSize, extRowSize, extEdgeSize, MPI_INT, &MPI_MAT_INT);
        MPI_Type_vector(extColSize, extRowSize, extEdgeSize, MPI_FLOAT, &MPI_MAT_FLOAT);
        MPI_Type_create_resized(MPI_MAT_INT, 0, sizeof(int), &MPI_MAT_INT);
        MPI_Type_create_resized(MPI_MAT_FLOAT, 0, sizeof(float), &MPI_MAT_FLOAT);
        MPI_Type_commit(&MPI_MAT_INT);
        MPI_Type_commit(&MPI_MAT_FLOAT);
        
        for (int i = 0; i < p_rows; i++) {
            for (int j = 0; j < p_cols; j++) {
                displs[i * p_cols + j] =  i * colSize * extEdgeSize  + j * rowSize;
            }
        }
    }
    
    MPI_Type_vector(2, rowSize, extRowSize, MPI_FLOAT, &MPI_TWO_ROW_FLOAT);
    MPI_Type_vector(colSize, 2, extRowSize, MPI_FLOAT, &MPI_TWO_COL_FLOAT);
    MPI_Type_vector(colSize, rowSize, extRowSize, MPI_FLOAT, &MPI_SUBMAT_FLOAT);
    MPI_Type_vector(colSize, rowSize, edgeSize, MPI_FLOAT, &MPI_SUBMAT_RECV_FLOAT);
    MPI_Type_create_resized(MPI_TWO_ROW_FLOAT, 0, sizeof(float), &MPI_TWO_ROW_FLOAT);
    MPI_Type_create_resized(MPI_TWO_COL_FLOAT, 0, sizeof(float), &MPI_TWO_COL_FLOAT);
    MPI_Type_create_resized(MPI_SUBMAT_FLOAT, 0, sizeof(float), &MPI_SUBMAT_FLOAT);
    MPI_Type_create_resized(MPI_SUBMAT_RECV_FLOAT, 0, sizeof(float), &MPI_SUBMAT_RECV_FLOAT);
    MPI_Type_commit(&MPI_TWO_ROW_FLOAT);
    MPI_Type_commit(&MPI_TWO_COL_FLOAT);
    MPI_Type_commit(&MPI_SUBMAT_FLOAT);
    MPI_Type_commit(&MPI_SUBMAT_RECV_FLOAT);
    
    MPI_Type_vector(extColSize, extRowSize, extRowSize, MPI_INT, &MPI_BLOCK_INT);
    MPI_Type_vector(extColSize, extRowSize, extRowSize, MPI_FLOAT, &MPI_BLOCK_FLOAT);
    MPI_Type_create_resized(MPI_BLOCK_INT, 0, sizeof(int), &MPI_BLOCK_INT);
    MPI_Type_create_resized(MPI_BLOCK_FLOAT, 0, sizeof(float), &MPI_BLOCK_FLOAT);
    MPI_Type_commit(&MPI_BLOCK_INT);
    MPI_Type_commit(&MPI_BLOCK_FLOAT);

    MPI_Scatterv(domMapSend.data(), sendcounts, displs, MPI_MAT_INT,
                 domainMap, 1, MPI_BLOCK_INT,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(domParamsSend.data(), sendcounts, displs, MPI_MAT_FLOAT,
                 domainParams, 1, MPI_BLOCK_FLOAT,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(tempSend.data(), sendcounts, displs, MPI_MAT_FLOAT,
                 temp, 1, MPI_BLOCK_FLOAT,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(tempSend.data(), sendcounts, displs, MPI_MAT_FLOAT,
                 newTemp, 1, MPI_BLOCK_FLOAT,
                 0, MPI_COMM_WORLD);
    
    MPI_Type_free(&MPI_BLOCK_INT);
    MPI_Type_free(&MPI_BLOCK_FLOAT);
    
    if (!m_rank) {
        MPI_Type_free(&MPI_MAT_INT);
        MPI_Type_free(&MPI_MAT_FLOAT);
        for (int i = 0; i < p_rows; i++) {
            for (int j = 0; j < p_cols; j++) {
                displs[i * p_cols + j] =  i * colSize * edgeSize  + j * rowSize;
            }
        }
    }

    if (modeRMA) {
        MPI_Win_create(temp, subSize * sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &winTemp);
        MPI_Win_create(newTemp, subSize * sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &winNewTemp);
    }
}

ParallelHeatSolver::~ParallelHeatSolver()
{
    delete[] domainMap;
    delete[] domainParams;
    delete[] temp;
    delete[] newTemp;

    delete[] res;
    delete[] displs;
    delete[] sendcounts;
    
    MPI_Type_free(&MPI_TWO_ROW_FLOAT);
    MPI_Type_free(&MPI_TWO_COL_FLOAT);
    MPI_Type_free(&MPI_SUBMAT_FLOAT);
    MPI_Type_free(&MPI_SUBMAT_RECV_FLOAT);

    MPI_Comm_free(&MPI_COL_COMM);

    if (modeRMA) {
        MPI_Win_free(&winNewTemp);
        MPI_Win_free(&winTemp);
    }
}

void ParallelHeatSolver::RunSolver(std::vector<float, AlignedAllocator<float> > &outResult)
{
    // Updatetemp(...) method can be used to evaluate heat equation over 2D temp
    //                 in parallel (using OpenMP).
    // NOTE: This method might be inefficient when used for small temps such as 
    //       2xN or Nx2 (these might arise at edges of the temp)
    //       In this case ComputePoint may be called directly in loop.
    
    // ShouldPrintProgress(N) returns true if average temperature should be reported
    // by 0th process at Nth time step (using "PrintProgressReport(...)").
    
    // Finally "PrintFinalReport(...)" should be used to print final elapsed time and
    // average temperature in column.
    double time = 0;
    float middleColAvgTemp = 0.0f;
    float localSum;
    const unsigned numIterations = m_simulationProperties.GetNumIterations(); 
    MPI_Request srequest[4] = {MPI_REQUEST_NULL,
                               MPI_REQUEST_NULL,
                               MPI_REQUEST_NULL,
                               MPI_REQUEST_NULL};

    MPI_Request rrequest[4] = {MPI_REQUEST_NULL,
                               MPI_REQUEST_NULL,
                               MPI_REQUEST_NULL,
                               MPI_REQUEST_NULL};
    MPI_Status status[4];

    if (!m_rank) {
        time = MPI_Wtime();
        outResult.resize(edgeSize * edgeSize);
    }

    for (unsigned iter = 0; iter < numIterations; ++iter) {
        // 4. Compute new temperature for each point in the domain (except borders)
        // border temperatures should remain constant (plus our stencil is +/-2 points)
        if (modeRMA)
            MPI_Win_fence(0, winNewTemp);
        
        if (!isTop) {
            UpdateTile(temp, newTemp,
                       domainParams, domainMap,
                       leftPadding, 2,
                       extRowSize - leftPadding - rightPadding, 2,
                       extRowSize,
                       m_simulationProperties.GetAirFlowRate(),
                       m_materialProperties.GetCoolerTemp());
            if (modeRMA) {
                MPI_Put(newTemp + topSendPadding, 1, MPI_TWO_ROW_FLOAT, 
                        topRank, botRecvPadding, 1, MPI_TWO_ROW_FLOAT, winNewTemp);
            } else {
                MPI_Irecv(newTemp + topRecvPadding, 1, MPI_TWO_ROW_FLOAT,
                        topRank, 0, MPI_COMM_WORLD, &rrequest[0]);
                MPI_Isend(newTemp + topSendPadding, 1, MPI_TWO_ROW_FLOAT,
                        topRank, 0, MPI_COMM_WORLD, &srequest[0]);
            }
        }
        if (!isBottom) {
            UpdateTile(temp, newTemp,
                       domainParams, domainMap,
                       leftPadding, colSize,
                       extRowSize - leftPadding - rightPadding, 2,
                       extRowSize,
                       m_simulationProperties.GetAirFlowRate(),
                       m_materialProperties.GetCoolerTemp());
            if (modeRMA) {
                MPI_Put(newTemp + botSendPadding, 1, MPI_TWO_ROW_FLOAT, 
                        botRank, topRecvPadding, 1, MPI_TWO_ROW_FLOAT, winNewTemp);
            } else {
                MPI_Irecv(newTemp + botRecvPadding, 1, MPI_TWO_ROW_FLOAT,
                        botRank, 0, MPI_COMM_WORLD, &rrequest[1]);
                MPI_Isend(newTemp + botSendPadding, 1, MPI_TWO_ROW_FLOAT,
                        botRank, 0, MPI_COMM_WORLD, &srequest[1]);
            }
        }
        if (!isLeft) {
            UpdateTile(temp, newTemp,
                       domainParams, domainMap,
                       2, 4,
                       2, extColSize - 8,
                       extRowSize,
                       m_simulationProperties.GetAirFlowRate(),
                       m_materialProperties.GetCoolerTemp());
            if (modeRMA) {
                MPI_Put(newTemp + leftSendPadding, 1, MPI_TWO_COL_FLOAT, 
                        leftRank, rightRecvPadding, 1, MPI_TWO_COL_FLOAT, winNewTemp);
            } else {
                MPI_Irecv(newTemp + leftRecvPadding, 1, MPI_TWO_COL_FLOAT,
                        leftRank, 0, MPI_COMM_WORLD, &rrequest[2]);
                MPI_Isend(newTemp + leftSendPadding, 1, MPI_TWO_COL_FLOAT,
                        leftRank, 0, MPI_COMM_WORLD, &srequest[2]);
            }
        }
        if (!isRight) {
            UpdateTile(temp, newTemp,
                       domainParams, domainMap,
                       rowSize, 4,
                       2, extColSize - 8,
                       extRowSize,
                       m_simulationProperties.GetAirFlowRate(),
                       m_materialProperties.GetCoolerTemp());
            if (modeRMA) {
                MPI_Put(newTemp + rightSendPadding, 1, MPI_TWO_COL_FLOAT, 
                        rightRank, leftRecvPadding, 1, MPI_TWO_COL_FLOAT, winNewTemp);
            } else {
                MPI_Irecv(newTemp + rightRecvPadding, 1, MPI_TWO_COL_FLOAT,
                        rightRank, 0, MPI_COMM_WORLD, &rrequest[3]);
                MPI_Isend(newTemp + rightSendPadding, 1, MPI_TWO_COL_FLOAT,
                        rightRank, 0, MPI_COMM_WORLD, &srequest[3]);
            }
        }

        UpdateTile(temp, newTemp,
                   domainParams, domainMap,
                   4, 4,
                   extRowSize - 8, extColSize - 8,
                   extRowSize,
                   m_simulationProperties.GetAirFlowRate(),
                   m_materialProperties.GetCoolerTemp());

        // 5. Compute average temperature in the middle column of the domain.
        if (isMiddle && ShouldPrintProgress(iter)) {
            localSum = 0.f;
            for (int i = 0; i < colSize; i++) {
                localSum += newTemp[(PADDING + i) * extRowSize + PADDING + middleCol];
            }
            MPI_Reduce(&localSum, &middleColAvgTemp, 1, MPI_FLOAT,
                       MPI_SUM, 0, MPI_COL_COMM);
            if (m_rank < p_cols) {
                cout << fixed;
                cout << "Progress " << setw(3) << unsigned(((iter + 1) * 100) / numIterations) <<
                        "% (Average Temperature " << fixed << middleColAvgTemp / edgeSize << " degrees)\n";
                cout << defaultfloat;
            }
        }
        
        // 6. Store the simulation state if appropriate (ie. every N-th iteration)
        if (write && ((iter % diskWriteIntensity) == 0)) {
            if (m_simulationProperties.IsUseParallelIO()) {
                parallelStoreDataIntoFile(iter);
            } else {
                MPI_Gatherv(newTemp + PADDING * extRowSize + PADDING, 1, MPI_SUBMAT_FLOAT,
                            res, sendcounts, displs, MPI_SUBMAT_RECV_FLOAT,
                            0, MPI_COMM_WORLD);
                if (!m_rank)
                    StoreDataIntoFile(m_fileHandle, iter, res);
            }
        }

        // 7. Swap source and destination buffers
        if (modeRMA) {
            MPI_Win_fence(0, winNewTemp);
            swap(winTemp, winNewTemp);
        } else {
            MPI_Waitall(4, rrequest, status);
        }
        swap(temp, newTemp);
    }
    
    MPI_Gatherv(temp + PADDING * extRowSize + PADDING, 1, MPI_SUBMAT_FLOAT,
                outResult.data(), sendcounts, displs, MPI_SUBMAT_RECV_FLOAT,
                0, MPI_COMM_WORLD);
    if (isMiddle) {
        localSum = 0.f;
        for (int i = 0; i < colSize; i++)
            localSum += temp[(PADDING + i) * extRowSize + PADDING + middleCol];

        MPI_Reduce(&localSum, &middleColAvgTemp, 1, MPI_FLOAT,
                   MPI_SUM, 0, MPI_COL_COMM);
        if (m_rank < p_cols) {
            MPI_Isend(&middleColAvgTemp, 1, MPI_FLOAT,
                      0, 1, MPI_COMM_WORLD, &srequest[0]);
        }
    }

    if (!m_rank) {
        MPI_Irecv(&middleColAvgTemp, 1, MPI_FLOAT,
                  p_cols / 2, 1, MPI_COMM_WORLD, &rrequest[0]);
        MPI_Wait(rrequest, status);
        time = MPI_Wtime() - time;
        PrintFinalReport(time, middleColAvgTemp / edgeSize, "par");
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

void ParallelHeatSolver::parallelStoreDataIntoFile(int iteration)
{
    // Create XFER property list and set Collective IO.
    hid_t xferPList = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(xferPList, H5FD_MPIO_COLLECTIVE);

    hsize_t gridSize[] = { hsize_t(edgeSize), hsize_t(edgeSize) };
    hsize_t memSize[] = { 1, hsize_t(rowSize) };

    hid_t filespace = H5Screate_simple(2, gridSize, nullptr);
    hid_t memspace  = H5Screate_simple(2, memSize,  nullptr);


    // 1. Create new HDF5 file group named as "Timestep_N", where "N" is number
    //    of current snapshot. The group is placed into root of the file "/Timestep_N".
    string groupName = "Timestep_" + to_string(static_cast<unsigned long long>(iteration / diskWriteIntensity));
    AutoHandle<hid_t> groupHandle(H5Gcreate(m_fileHandle, groupName.c_str(),
                                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose);
    // NOTE: AutoHandle<hid_t> is object wrapping HDF5 handle so that its automatically
    //       released when object goes out of scope (see RAII pattern).
    //       The class can be found in "base.h".

    {
        // 2. Create new dataset "/Timestep_N/Temperature" which is simulation-domain
        //    sized 2D array of "float"s.
        string dataSetName("Temperature");
        // 2.1 Define shape of the dataset (2D edgeSize x edgeSize array).
        AutoHandle<hid_t> dataSpaceHandle(H5Screate_simple(2, gridSize, NULL), H5Sclose);
        // 2.2 Create datased with specified shape.
        AutoHandle<hid_t> dataSetHandle(H5Dcreate(groupHandle, dataSetName.c_str(),
                                                  H5T_NATIVE_FLOAT, dataSpaceHandle,
                                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Dclose);


        // Write the data from memory pointed by "data" into new datased.
        // Note that we are filling whole dataset and therefore we can specify
        // "H5S_ALL" for both memory and dataset spaces.
        
        hsize_t slabStart[] = {hsize_t((m_rank / p_cols) * colSize),
                               hsize_t(m_rank % p_cols) * rowSize};
        for (int i = 0; i < colSize; i++) {
            H5Sselect_hyperslab(
                    filespace,
                    H5S_SELECT_SET,
                    slabStart,
                    nullptr,
                    memSize,
                    nullptr);

            H5Dwrite(
                    dataSetHandle,
                    H5T_NATIVE_FLOAT,
                    memspace,
                    filespace,
                    xferPList,
                    newTemp + extRowSize * (i + PADDING) + PADDING);
            
            slabStart[0] += 1;
        }
        // NOTE: Both dataset and dataspace will be closed here automatically (due to RAII).
    }
    {
        // 3. Create Integer attribute in the same group "/Timestep_N/Time"
        //    in which we store number of current simulation iteration.
        std::string attributeName("Time");

        // 3.1 Dataspace is single value/scalar.
        AutoHandle<hid_t> dataSpaceHandle(H5Screate(H5S_SCALAR), H5Sclose);

        // 3.2 Create the attribute in the group as double.
        AutoHandle<hid_t> attributeHandle(H5Acreate2(groupHandle, attributeName.c_str(),
                                                     H5T_IEEE_F64LE, dataSpaceHandle,
                                                     H5P_DEFAULT, H5P_DEFAULT), H5Aclose);

        // 3.3 Write value into the attribute.
        double snapshotTime = double(iteration);
        H5Awrite(attributeHandle, H5T_IEEE_F64LE, &snapshotTime);

        // NOTE: Both dataspace and attribute handles will be released here.
    }
    // NOTE: The group handle will be released here.
    H5Pclose(xferPList);
}
