/*
 * Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "stdioInterf.h"
#include "fioMacros.h"

#include <memory.h>

#include "fort_vars.h"

extern void *shmalloc(size_t);

/* ==================== local heap routines ====================== */

#define ZIP ((char *)15L)

void *sh_alloc(int id, size_t n);
void sh_free(void *ptr);
void *sh_realloc(int id, void *ptr, size_t size);
void *ben_calloc(int id, size_t nmemb, size_t size);

/* malloc */

void *
__fort_malloc_without_abort(size_t n)
{
  char *p;

  if (n == 0)
    return ZIP;
  p = malloc(n);
  if (__fort_zmem && (p != NULL))
    memset(p, '\0', n);
  return p;
}

void *
__sh_fort_malloc_without_abort(int id, size_t n)
{
  char *p;

  if (n == 0)
    return ZIP;
  p = sh_alloc(id, n);
  if (__fort_zmem && (p != NULL))
    memset(p, '\0', n);
  return p;
}

void *
__fort_malloc(size_t n)
{
  char *p;

  p = __fort_malloc_without_abort(n);
  if (p == (char *)0)
    __fort_abort("__fort_malloc: not enough memory");
  return p;
}

void *
__sh_fort_malloc(int id, size_t n)
{
  char *p;

  p = __sh_fort_malloc_without_abort(id, n);
  if (p == (char *)0)
    __fort_abort("__fort_malloc: not enough memory");
  return p;
}

/* realloc */

void *
__fort_realloc(void *ptr, size_t n)
{
  char *p;

  if (ptr == (char *)0 | ptr == ZIP) {
    if (n == 0)
      return ZIP;
    p = malloc(n);
    if (__fort_zmem && (p != NULL))
      memset(p, '\0', n);
  } else {
    if (n == 0) {
      free(ptr);
      return ZIP;
    }
    p = realloc(ptr, n);
  }
  if (p == (char *)0) {
    __fort_abort("__fort_realloc: not enough memory");
  }
  return (p);
}

void *
__sh_fort_realloc(int id, void *ptr, size_t n)
{
  char *p;

  if (ptr == (char *)0 | ptr == ZIP) {
    if (n == 0)
      return ZIP;
    p = sh_alloc(id, n);
    if (__fort_zmem && (p != NULL))
      memset(p, '\0', n);
  } else {
    if (n == 0) {
      sh_free(ptr);
      return ZIP;
    }
    p = sh_realloc(id, ptr, n);
  }
  if (p == (char *)0) {
    __fort_abort("__fort_realloc: not enough memory");
  }
  return (p);
}

/* calloc */

void *
__fort_calloc_without_abort(size_t n)
{
  char *p;

  if (n == 0)
    return ZIP;
  p = malloc(n);
  if (p != NULL)
    memset(p, '\0', n);
  return p;
}

void *
__sh_fort_calloc_without_abort(int id, size_t n)
{
  char *p;

  if (n == 0)
    return ZIP;
  p = sh_alloc(id, n);
  if (p != NULL)
    memset(p, '\0', n);
  return p;
}

void *
__fort_calloc(size_t n, size_t s)
{
  char *p;

  if (n == 0 | s == 0)
    return ZIP;
  p = calloc(n, s);
  if (p == (char *)0) {
    __fort_abort("__fort_calloc: not enough memory");
  }
  return (p);
}

void *
__sh_fort_calloc(int id, size_t n, size_t s)
{
  char *p;

  if (n == 0 | s == 0)
    return ZIP;
  p = sh_calloc(id, n, s);
  if (p == (char *)0) {
    __fort_abort("__fort_calloc: not enough memory");
  }
  return (p);
}

/* free */

void
__fort_free(void *ptr)
{
  if (ptr != (char *)0 & ptr != ZIP) {
    free(ptr);
  }
}

void
__sh_fort_free(void *ptr)
{
  if (ptr != (char *)0 & ptr != ZIP) {
    sh_free(ptr);
  }
}

/* ================= pseudo-global heap routines ================= */

/* stubs for global shared memory (mmapped) allocation calls */

void *
__fort_gmalloc_without_abort(size_t n)
{
  return __fort_malloc_without_abort(n);
}

void *
__sh_fort_gmalloc_without_abort(int id, size_t n)
{
  return __sh_fort_malloc_without_abort(id, n);
}

void *
__fort_gmalloc(size_t n)
{
  return __fort_malloc(n);
}

void *
__sh_fort_gmalloc(int id, size_t n)
{
  return __sh_fort_malloc(id, n);
}

void *
__fort_grealloc(void *ptr, size_t n)
{
  return __fort_realloc(ptr, n);
}

void *
__sh_fort_grealloc(int id, void *ptr, size_t n)
{
  return __sh_fort_realloc(id, ptr, n);
}

void *
__fort_gcalloc_without_abort(size_t n)
{
  return __fort_calloc_without_abort(n);
}

void *
__sh_fort_gcalloc_without_abort(int id, size_t n)
{
  return __sh_fort_calloc_without_abort(id, n);
}

void *
__fort_gcalloc(size_t n, size_t s)
{
  return __fort_calloc(n, s);
}

void *
__sh_fort_gcalloc(int id, size_t n, size_t s)
{
  return __sh_fort_calloc(id, n, s);
}

void
__fort_gfree(void *ptr)
{
  __fort_free(ptr);
}

void
__sh_fort_gfree(void *ptr)
{
  __sh_fort_free(ptr);
}
