#ifndef BITBOARD_H
#define BITBOARD_H

#include "state.h"
#include <omp.h>

/*
    data
*/
extern BITBOARD KnightMoves[64],  
                KingMoves[64],
                PawnAttacksBlack[64],
                PawnAttacksWhite[64],
                PawnMovesBlack[64],
                PawnMovesWhite[64];
#pragma omp threadprivate(KnightMoves, KingMoves, PawnAttacksBlack, PawnAttacksWhite, PawnMovesBlack, PawnMovesWhite)
extern BITBOARD Mask[64], 
                InvMask[64];
#pragma omp threadprivate(Mask, InvMask)
                
extern BITBOARD DiagMaska1h8[64], 
                DiagMaska8h1[64];
#pragma omp threadprivate(DiagMaska1h8, DiagMaska8h1)
extern BITBOARD FileMask[8],
                RankMask[8],
                AboveMask[8],
                BelowMask[8],
                LeftMask[8],
                RightMask[8];
#pragma omp threadprivate(FileMask, RankMask, AboveMask, BelowMask, LeftMask, RightMask)
extern BITBOARD RookMask[64],
                BishopMask[64],
                QueenMask[64]; 
#pragma omp threadprivate(RookMask, BishopMask, QueenMask)
extern BITBOARD CastleMask[4];
#pragma omp threadprivate(CastleMask)
extern BITBOARD FileUpMask[64],
                FileDownMask[64];
#pragma omp threadprivate(FileUpMask, FileDownMask)
extern BITBOARD WhiteKingSide, 
                WhiteQueenSide, 
                BlackKingSide,  
                BlackQueenSide;
#pragma omp threadprivate(WhiteKingSide, WhiteQueenSide, BlackKingSide, BlackQueenSide)
extern BITBOARD KingSafetyMask[64],
                KingSafetyMask1[64];
#pragma omp threadprivate(KingSafetyMask, KingSafetyMask1)
extern BITBOARD WhiteStrongSquareMask,
                BlackStrongSquareMask;
#pragma omp threadprivate(WhiteStrongSquareMask, BlackStrongSquareMask)
extern BITBOARD WhiteSqMask,
                BlackSqMask;
#pragma omp threadprivate(WhiteSqMask, BlackSqMask)
extern BITBOARD KSMask,
                QSMask;                    
#pragma omp threadprivate(KSMask, QSMask)
extern BITBOARD KingFilesMask[8],
                KingPressureMask[64],
                KingPressureMask1[64],
                CenterMask,
                SpaceMask[2];   
#pragma omp threadprivate(KingFilesMask, KingPressureMask, KingPressureMask1, CenterMask, SpaceMask)
/*
    Alias array notation to more expressive notation
*/
#define BlackPawns BitBoard[bpawn]
#define WhitePawns BitBoard[wpawn]
#define BlackRooks BitBoard[brook]
#define WhiteRooks BitBoard[wrook]
#define BlackKnights BitBoard[bknight]
#define WhiteKnights BitBoard[wknight]
#define BlackBishops BitBoard[bbishop]
#define WhiteBishops BitBoard[wbishop]
#define BlackQueens BitBoard[bqueen]
#define WhiteQueens BitBoard[wqueen]
#define BlackKing BitBoard[bking]
#define WhiteKing BitBoard[wking]

/* 
    functions
*/
void PrintBitboard(const BITBOARD B);
void PrintAllBitboards(state_t *s);
void SetupPrecalculatedData(void);
void SetupCastleMasks(state_t *g);

#if !defined(SPEC)
const BITBOARD RankAttacks(const BITBOARD occ, const unsigned int sq);
const BITBOARD FileAttacks(BITBOARD occ, const unsigned int sq);
const BITBOARD DiagonalAttacks(BITBOARD occ, const unsigned int sq);
const BITBOARD AntiDiagAttacks(BITBOARD occ, const unsigned int sq);
#else
BITBOARD RankAttacks(const BITBOARD occ, const unsigned int sq);
BITBOARD FileAttacks(BITBOARD occ, const unsigned int sq);
BITBOARD DiagonalAttacks(BITBOARD occ, const unsigned int sq);
BITBOARD AntiDiagAttacks(BITBOARD occ, const unsigned int sq);
#endif
BITBOARD RookAttacks(state_t *s, const int sq);
BITBOARD BishopAttacks(state_t *s, const int sq);

#endif

