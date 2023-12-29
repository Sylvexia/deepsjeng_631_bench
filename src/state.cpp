/*
    Sjeng - a chess playing program
    Copyright (C) 2000-2008 Gian-Carlo Pascutto

    File: state.c
    Purpose: state structures                  
*/

#include "config.h"
#include "sjeng.h"
#include "state.h"
#include <omp.h>

/* 
    global state
*/
state_t           state;
gamestate_t   gamestate;
scoreboard_t scoreboard;
#pragma omp threadprivate(scoreboard)