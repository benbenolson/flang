! Copyright (c) 2011, NVIDIA CORPORATION.  All rights reserved.
!
! Licensed under the Apache License, Version 2.0 (the "License");
! you may not use this file except in compliance with the License.
! You may obtain a copy of the License at
!
!     http://www.apache.org/licenses/LICENSE-2.0
!
! Unless required by applicable law or agreed to in writing, software
! distributed under the License is distributed on an "AS IS" BASIS,
! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
! See the License for the specific language governing permissions and
! limitations under the License.
!

module mod_gen
implicit none
private
type, public :: v
real, allocatable :: r(:)
end type
end module

program p
USE CHECK_MOD
use mod_gen
class(v),allocatable ::stuff
real, allocatable :: r(:)
class(v),allocatable :: stuff2
real :: rr(10)
logical expect(4)
logical rslt(4)

expect = .true.
rslt = .false.

allocate(stuff)
allocate(r(10))

do i=1,10
r(i) = i
rr(i) = i
enddo

allocate(stuff%r(size(rr)),source=r)
!print *, allocated(stuff%r)
rslt(1) = allocated(stuff%r)
allocate(stuff2,source=stuff)
deallocate(stuff%r)
!print *,allocated(stuff2%r),allocated(stuff%r)
rslt(2) = allocated(stuff2%r)
rslt(3) = .not. allocated(stuff%r)
!print *, stuff2
rslt(4) = all(stuff2%r .eq. rr)

call check(rslt,expect,4)

end
