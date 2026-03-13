!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2025 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Driver mode - To communicate with i-PI Python wrapper
!> \par History
!>      none
!> \author Michele Ceriotti 03.2012
!> \note Modified to support extras JSON output including fermi_level_eV and charge
! **************************************************************************************************
MODULE ipi_driver
   USE ISO_C_BINDING, ONLY: C_CHAR, &
                            C_DOUBLE, &
                            C_INT, &
                            C_LOC, &
                            C_NULL_CHAR, &
                            C_PTR
   USE bibliography, ONLY: Ceriotti2014, &
                           Kapil2016, &
                           cite_reference
   USE cell_methods, ONLY: cell_create, &
                           init_cell
   USE cell_types, ONLY: cell_release, &
                         cell_type
   USE cp_external_control, ONLY: external_control
   USE cp_log_handling, ONLY: cp_logger_get_default_io_unit
   USE cp_subsys_types, ONLY: cp_subsys_get, &
                              cp_subsys_set, &
                              cp_subsys_type
   USE force_env_methods, ONLY: force_env_calc_energy_force
   USE force_env_types, ONLY: force_env_get, &
                              force_env_type
   USE global_types, ONLY: global_environment_type
   USE input_section_types, ONLY: section_vals_get_subs_vals, &
                                  section_vals_type, &
                                  section_vals_val_get
   USE kinds, ONLY: default_path_length, &
                    default_string_length, &
                    dp, &
                    int_4
   USE message_passing, ONLY: mp_para_env_type, &
                              mp_request_type, &
                              mp_testany
   ! Additional USE statements for extras JSON support
   USE qs_environment_types, ONLY: get_qs_env, &
                                   qs_environment_type
   USE qs_kind_types, ONLY: qs_kind_type, &
                            get_qs_kind
   USE external_potential_types, ONLY: all_potential_type, &
                                       gth_potential_type, &
                                       sgp_potential_type
   USE qs_mo_types, ONLY: get_mo_set, &
                          mo_set_type
   USE qs_energy_types, ONLY: qs_energy_type
   USE atomic_kind_types, ONLY: atomic_kind_type, &
                                get_atomic_kind
   USE particle_types, ONLY: particle_type
#ifndef __NO_SOCKETS
   USE sockets_interface, ONLY: writebuffer, &
                                readbuffer, &
                                open_connect_socket, &
                                uwait
#endif
   USE virial_types, ONLY: virial_type
#include "./base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'ipi_driver'

   ! Unit conversion constants
   REAL(KIND=dp), PARAMETER :: HARTREE_TO_EV = 27.211386245988_dp

   PUBLIC :: run_driver

CONTAINS

! **************************************************************************************************
!> \brief Get fermi level and charge for extras JSON
!> \param force_env The force environment
!> \param extras_json JSON string with extras information
!> \author Assistant 2025
! **************************************************************************************************
   SUBROUTINE get_extras_json(force_env, extras_json)
      TYPE(force_env_type), POINTER                      :: force_env
      CHARACTER(LEN=*), INTENT(OUT)                      :: extras_json

      CHARACTER(len=*), PARAMETER :: routineN = 'get_extras_json'

      INTEGER :: handle
      TYPE(qs_environment_type), POINTER                 :: qs_env
      TYPE(qs_energy_type), POINTER                      :: energy
      TYPE(section_vals_type), POINTER                   :: dft_section
      TYPE(mo_set_type), DIMENSION(:), POINTER           :: mos

      REAL(KIND=dp)                                      :: fermi_level_eV, fermi_hartree
      LOGICAL                                            :: qs_method
      INTEGER                                            :: charge_int, nspin, ispin
      CHARACTER(LEN=20)                                  :: fermi_str, charge_str

      ! Initialize defaults
      CALL timeset(routineN, handle)
      fermi_level_eV = 0.0_dp
      charge_int = 0

      ! Check if this is a QS calculation
      NULLIFY(qs_env)
      CALL force_env_get(force_env, qs_env=qs_env)
      qs_method = ASSOCIATED(qs_env)

      IF (qs_method) THEN
         ! Get Fermi energy directly from qs_energy
         NULLIFY(energy)
         NULLIFY(mos)
         CALL get_qs_env(qs_env, energy=energy, mos=mos)

         IF (ASSOCIATED(energy)) THEN
            ! Get fermi_energy from energy structure
            fermi_hartree = energy%efermi

            ! If efermi is exactly zero, recompute it from the MO sets as
            ! the average of mos(ispin)%mu over spin channels.
            IF (fermi_hartree == 0.0_dp) THEN
               IF (ASSOCIATED(mos)) THEN
                  nspin = SIZE(mos)
                  IF (nspin > 0) THEN
                     fermi_hartree = 0.0_dp
                     DO ispin = 1, nspin
                        fermi_hartree = fermi_hartree + mos(ispin)%mu
                     END DO
                     fermi_hartree = fermi_hartree / REAL(nspin, KIND=dp)
                  END IF
               END IF
            END IF

            ! Convert from Hartree to eV
            fermi_level_eV = fermi_hartree * HARTREE_TO_EV
         ELSE
            fermi_level_eV = 0.0_dp
         END IF

         ! Get charge from input section - use force_env input
         NULLIFY(dft_section)
         CALL force_env_get(force_env, input=dft_section)
         IF (ASSOCIATED(dft_section)) THEN
            dft_section => section_vals_get_subs_vals(dft_section, "DFT")
            IF (ASSOCIATED(dft_section)) THEN
               CALL section_vals_val_get(dft_section, "CHARGE", i_val=charge_int)
            END IF
         END IF

      END IF

      ! Format the JSON string
      WRITE(fermi_str, '(F12.6)') fermi_level_eV
      WRITE(charge_str, '(I0)') charge_int
      ! Construct JSON - keep it simple for robustness
      extras_json = '{"fermi_level_eV": ' // TRIM(ADJUSTL(fermi_str)) // &
                   ', "charge": ' // TRIM(ADJUSTL(charge_str)) // '}'

      CALL timestop(handle)
   END SUBROUTINE get_extras_json
! **************************************************************************************************
!> \brief ...
!> \param force_env ...
!> \param globenv ...
!> \par History
!>       12.2013 included in repository
!> \author Ceriotti
! **************************************************************************************************

   SUBROUTINE run_driver(force_env, globenv)
      TYPE(force_env_type), POINTER            :: force_env
      TYPE(global_environment_type), POINTER   :: globenv

      CHARACTER(len=*), PARAMETER :: routineN = 'run_driver'

#ifdef __NO_SOCKETS
      INTEGER                                  :: handle
      CALL timeset(routineN, handle)
      CPABORT("CP2K was compiled with the __NO_SOCKETS option!")
      MARK_USED(globenv)
      MARK_USED(force_env)
#else
      INTEGER, PARAMETER                       :: MSGLEN = 12

      CHARACTER(len=default_path_length)       :: c_hostname, drv_hostname
      CHARACTER(LEN=default_string_length)     :: header
      CHARACTER(LEN=1000)                      :: extras_json  ! Buffer for JSON extras
      INTEGER                                  :: drv_port, handle, i_drv_unix, &
                                                  idir, ii, inet, ip, iwait, &
                                                  nat, output_unit, socket, &
                                                  json_len
      TYPE(mp_request_type), DIMENSION(2) ::                                            wait_req
      INTEGER(KIND=int_4), POINTER             :: wait_msg(:)
      LOGICAL                                  :: drv_unix, fwait, hasdata, &
                                                  ionode, should_stop
      REAL(KIND=dp)                            :: cellh(3, 3), cellih(3, 3), &
                                                  mxmat(9), pot, vir(3, 3)
      REAL(KIND=dp), ALLOCATABLE               :: combuf(:)
      TYPE(cell_type), POINTER                 :: cpcell
      TYPE(mp_para_env_type), POINTER          :: para_env
      TYPE(cp_subsys_type), POINTER            :: subsys
      TYPE(section_vals_type), POINTER         :: drv_section, motion_section
      TYPE(virial_type), POINTER               :: virial
      REAL(KIND=dp)                            :: sleeptime, core_corr_ne

      CALL timeset(routineN, handle)

      CALL cite_reference(Ceriotti2014)
      CALL cite_reference(Kapil2016)

! server address parsing
! buffers and temporaries for communication
! access cp2k structures

      CPASSERT(ASSOCIATED(force_env))
      CALL force_env_get(force_env, para_env=para_env)

      hasdata = .FALSE.
      ionode = para_env%is_source()

      output_unit = cp_logger_get_default_io_unit()

      ! reads driver parameters from input
      motion_section => section_vals_get_subs_vals(force_env%root_section, "MOTION")
      drv_section => section_vals_get_subs_vals(motion_section, "DRIVER")

      CALL section_vals_val_get(drv_section, "HOST", c_val=drv_hostname)
      CALL section_vals_val_get(drv_section, "PORT", i_val=drv_port)
      CALL section_vals_val_get(drv_section, "UNIX", l_val=drv_unix)
      CALL section_vals_val_get(drv_section, "SLEEP_TIME", r_val=sleeptime)
      CPASSERT(sleeptime >= 0)

      ! opens the socket
      socket = 0
      inet = 1
      i_drv_unix = 1 ! a bit convoluted. socket.c uses a different convention...
      IF (drv_unix) i_drv_unix = 0
      IF (output_unit > 0) THEN
         WRITE (output_unit, *) "@ i-PI DRIVER BEING LOADED"
         WRITE (output_unit, *) "@ INPUT DATA: ", TRIM(drv_hostname), drv_port, drv_unix
      END IF

      c_hostname = TRIM(drv_hostname)//C_NULL_CHAR
      IF (ionode) CALL open_connect_socket(socket, i_drv_unix, drv_port, c_hostname)

      NULLIFY (wait_msg)
      ALLOCATE (wait_msg(1))
      !now we have a socket, so we can initialize the CP2K environments.
      NULLIFY (cpcell)
      CALL cell_create(cpcell)
      driver_loop: DO
         ! do communication on master node only...
         header = ""

         CALL para_env%sync()

         ! non-blocking sync to avoid useless CPU consumption
         IF (ionode) THEN
            CALL readbuffer(socket, header, MSGLEN)
            wait_msg = 0
            DO iwait = 0, para_env%num_pe - 1
               IF (iwait /= para_env%source) THEN
                  CALL para_env%send(msg=wait_msg, dest=iwait, tag=666)
               END IF
            END DO
         ELSE
            CALL para_env%irecv(msgout=wait_msg, source=para_env%source, &
                                tag=666, request=wait_req(2))
            CALL mp_testany(wait_req(2:), flag=fwait)
            DO WHILE (.NOT. fwait)
               CALL mp_testany(wait_req(2:), flag=fwait)
               CALL uwait(sleeptime)
            END DO
         END IF

         CALL para_env%sync()

         CALL para_env%bcast(header)

         IF (output_unit > 0) WRITE (output_unit, *) " @ DRIVER MODE: Message from server: ", TRIM(header)

         IF (TRIM(header) == "STATUS") THEN
            CALL para_env%sync()
            IF (ionode) THEN ! does not  need init (well, maybe it should, just to check atom numbers and the like... )
               IF (hasdata) THEN
                  CALL writebuffer(socket, "HAVEDATA    ", MSGLEN)
               ELSE
                  CALL writebuffer(socket, "READY       ", MSGLEN)
               END IF
            END IF
            CALL para_env%sync()
         ELSE IF (TRIM(header) == "POSDATA") THEN
            IF (ionode) THEN
               CALL readbuffer(socket, mxmat, 9)
               cellh = RESHAPE(mxmat, (/3, 3/))
               CALL readbuffer(socket, mxmat, 9)
               cellih = RESHAPE(mxmat, (/3, 3/))
               CALL readbuffer(socket, nat)
               cellh = TRANSPOSE(cellh)
               cellih = TRANSPOSE(cellih)
            END IF
            CALL para_env%bcast(cellh, 0)
            CALL para_env%bcast(cellih, 0)
            CALL para_env%bcast(nat, 0)
            IF (.NOT. ALLOCATED(combuf)) ALLOCATE (combuf(3*nat))
            IF (ionode) CALL readbuffer(socket, combuf, nat*3)
            CALL para_env%bcast(combuf, 0)

            CALL force_env_get(force_env, subsys=subsys)
            IF (nat /= subsys%particles%n_els) &
               CPABORT("@DRIVER MODE: Uh-oh! Particle number mismatch between i-PI and cp2k input!")
            ii = 0
            DO ip = 1, subsys%particles%n_els
               DO idir = 1, 3
                  ii = ii + 1
                  subsys%particles%els(ip)%r(idir) = combuf(ii)
               END DO
            END DO
            CALL init_cell(cpcell, hmat=cellh)
            CALL cp_subsys_set(subsys, cell=cpcell)

            CALL force_env_calc_energy_force(force_env, calc_force=.TRUE.)

            IF (output_unit > 0) WRITE (output_unit, *) " @ DRIVER MODE: Received positions "

            combuf = 0
            ii = 0
            DO ip = 1, subsys%particles%n_els
               DO idir = 1, 3
                  ii = ii + 1
                  combuf(ii) = subsys%particles%els(ip)%f(idir)
               END DO
            END DO
            CALL force_env_get(force_env, potential_energy=pot)
            CALL force_env_get(force_env, cell=cpcell)
            CALL cp_subsys_get(subsys, virial=virial)

            vir = TRANSPOSE(virial%pv_virial)

            CALL external_control(should_stop, "IPI", globenv=globenv)
            IF (should_stop) EXIT

            hasdata = .TRUE.
         ELSE IF (TRIM(header) == "GETFORCE") THEN
            IF (output_unit > 0) WRITE (output_unit, *) " @ DRIVER MODE: Returning v,forces,stress "
            IF (ionode) THEN
               CALL writebuffer(socket, "FORCEREADY  ", MSGLEN)
               CALL writebuffer(socket, pot)
               CALL writebuffer(socket, nat)
               CALL writebuffer(socket, combuf, 3*nat)
               CALL writebuffer(socket, RESHAPE(vir, (/9/)), 9)

               ! Get extras JSON information including fermi_level_eV and charge
               CALL get_extras_json(force_env, extras_json)
               json_len = LEN_TRIM(extras_json)

               IF (output_unit > 0) THEN
                  WRITE (output_unit, *) " @ DRIVER MODE: Sending extras JSON: ", TRIM(extras_json)
               END IF

               ! Send the length of the JSON string, then the string itself
               CALL writebuffer(socket, json_len)
               IF (json_len > 0) THEN
                  ! Send JSON string as a character buffer
                  CALL writebuffer(socket, TRIM(extras_json), json_len)
               END IF
            END IF
            hasdata = .FALSE.
         ELSE IF (TRIM(header) == "CHGDATA") THEN
            ! Lightweight charge update message carrying a single double
            ! precision value. In the CP2K Ne-doping constant-potential
            ! implementation this value represents the per-Ne core
            ! correction (CORE_CORRECTION) to be applied to all Ne atomic
            ! kinds in the QS subsystem. A positive value increases the
            ! effective nuclear charge correction for Ne.

            IF (ionode) THEN
               CALL readbuffer(socket, core_corr_ne)
            END IF

            CALL para_env%bcast(core_corr_ne, 0)

            ! Apply the updated core correction to all Ne atomic kinds. This
            ! is a local modification of the electronic structure and does
            ! not alter geometry, so it can be safely done outside the
            ! POSDATA/GETFORCE sequence. If anything goes wrong we abort,
            ! since a partially updated core correction state would be
            ! difficult to diagnose.
            CALL apply_ne_core_correction(force_env, core_corr_ne)
         ELSE
            IF (output_unit > 0) WRITE (output_unit, *) " @DRIVER MODE:  Socket disconnected, time to exit."
            EXIT
         END IF
      END DO driver_loop

      ! clean up
      CALL cell_release(cpcell)
      DEALLOCATE (wait_msg)
#endif

      CALL timestop(handle)

   END SUBROUTINE run_driver

! **************************************************************************************************
!> \brief   Apply Ne core correction update from i-PI.
!>
!> Receives a per-Ne core correction value core_corr_ne (in electrons) and
!> applies it to all Ne atomic kinds in the QS subsystem by updating the
!> underlying all-electron or GTH/SGP potential zeff_correction field. This
!> effectively corresponds to adjusting the CORE_CORRECTION keyword for Ne.
!>
!> The implementation traverses the QS environment, obtains the atomic kind
!> and QS kind arrays, and for each kind whose element symbol is 'Ne' it
!> updates zeff_correction := core_corr_ne in the associated potential data.
!> The resulting change is reflected in the subsequent charge and electron
!> counts through the existing zeff_correction usage in QS routines.
! **************************************************************************************************
   SUBROUTINE apply_ne_core_correction(force_env, core_corr_ne)

      TYPE(force_env_type), POINTER                      :: force_env
      REAL(KIND=dp), INTENT(IN)                          :: core_corr_ne

      CHARACTER(LEN=*), PARAMETER                        :: routineN = 'apply_ne_core_correction'

      TYPE(qs_environment_type), POINTER                 :: qs_env
      TYPE(atomic_kind_type), DIMENSION(:), POINTER      :: atomic_kind_set
      TYPE(qs_kind_type), DIMENSION(:), POINTER          :: qs_kind_set
      TYPE(atomic_kind_type), POINTER                    :: atomic_kind
      TYPE(qs_kind_type), POINTER                        :: qs_kind
      TYPE(all_potential_type), POINTER                  :: all_potential
      TYPE(gth_potential_type), POINTER                  :: gth_potential
      TYPE(sgp_potential_type), POINTER                  :: sgp_potential

      INTEGER                                            :: handle, ikind, nkind
      CHARACTER(LEN=2)                                   :: elem

      CALL timeset(routineN, handle)

      ! Obtain QS environment and atomic kind / kind sets. If this is not a
      ! QS calculation there is nothing to do.
      NULLIFY(qs_env)
      CALL force_env_get(force_env, qs_env=qs_env)
      IF (.NOT. ASSOCIATED(qs_env)) THEN
         CALL timestop(handle)
         RETURN
      END IF

      NULLIFY(atomic_kind_set)
      NULLIFY(qs_kind_set)
      CALL get_qs_env(qs_env=qs_env, atomic_kind_set=atomic_kind_set, qs_kind_set=qs_kind_set)

      IF (.NOT. (ASSOCIATED(atomic_kind_set) .AND. ASSOCIATED(qs_kind_set))) THEN
         CALL timestop(handle)
         RETURN
      END IF

      nkind = SIZE(atomic_kind_set, 1)

      DO ikind = 1, nkind
         atomic_kind => atomic_kind_set(ikind)
         qs_kind     => qs_kind_set(ikind)

         ! Identify Ne atomic kinds by their element symbol. Compare against
         ! the canonical two-character symbol "Ne" (and its fully uppercase
         ! variant) after left adjustment.
         elem = atomic_kind%element_symbol
         elem = ADJUSTL(elem)

         IF (TRIM(elem) == 'Ne' .OR. TRIM(elem) == 'NE') THEN
            NULLIFY(all_potential)
            NULLIFY(gth_potential)
            NULLIFY(sgp_potential)

            CALL get_qs_kind(qs_kind, all_potential=all_potential, gth_potential=gth_potential, &
                             sgp_potential=sgp_potential)

            IF (ASSOCIATED(all_potential)) THEN
               all_potential%zeff_correction = core_corr_ne
            END IF
            IF (ASSOCIATED(gth_potential)) THEN
               gth_potential%zeff_correction = core_corr_ne
            END IF
            IF (ASSOCIATED(sgp_potential)) THEN
               sgp_potential%zeff_correction = core_corr_ne
            END IF
         END IF
      END DO

      CALL timestop(handle)

   END SUBROUTINE apply_ne_core_correction

   END MODULE ipi_driver