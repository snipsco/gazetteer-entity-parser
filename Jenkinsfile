@Library('snips@0.2.5.4') _


targets = []

build = [
    stableBranch: "master",
    devBranch: "develop",
    target: "all",
    toolchainVersion: "0.3.0"
]


node('jenkins-slave-generic') {

   stage('build') {
        checkoutSources()
	targets = setupTargets(build) { build, target ->
            target.id = "${target.osType}_${target.osVersion}_${target.archType}"
            target.osCode = "${target.osType}-${target.osVersion}"
    	}

        build = setupBuild(build, {}, 'Cargo.toml')
   }

   parallel(
       "linux": { selectTargets(targets, { it.type == "linux" }) { linuxTargets ->

            parallelize(linuxTargets) { target ->

                machine(target, "build") {

                    section(target, "checkoutSources-${target.id}") {
                      deleteDir()
                      checkoutSources(build)
                    }

                    section(target, "build-${target.id}") {
                      ssh_sh 'cargo build'
                    }

                    section(target, "test-${target.id}") {
                      ssh_sh 'cargo test'
                    }

                }

            } // end paralle
         } //end linux
     }
   )

   // if (build.isRelease) {
   //      machine(targets, "release") {
   //          section(targets, "Release") {
   //              performReleaseIfNeeded(build)
   //          }
   //      }
   //  }
}
