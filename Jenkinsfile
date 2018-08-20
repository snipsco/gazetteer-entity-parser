@Library('snips@0.2.5.4') _


targets = []

build = [
    stableBranch: "master",
    devBranch: "develop",
    target: "all",
    toolchainVersion: "0.3.0"
]

targetNames = [ 'linux', 'macos', 'ios' ]

def jobs = [:]

node('jenkins-slave-generic') {

   stage('setup-build') {
        checkoutSources()
	targets = setupTargets(build) { build, target ->
            target.id = "${target.osType}_${target.osVersion}_${target.archType}"
            target.osCode = "${target.osType}-${target.osVersion}"
    	}
        build = setupBuild(build, {}, 'Cargo.toml')
   }

   targetNames.each{ targetName ->
           selectTargets(targets, { it.type == "${targetName}" }) { buildTargets ->

               jobs["${targetName}"] = { parallelize(buildTargets) { target ->

                   machine(target, "build") {
                       section(target, "checkoutSources-${target.id}") {
                           deleteDir()
                           checkoutSources(build)
                       }

                       section(target, "build-${target.id}") {
                            if (targetName != 'ios') {
                                ssh_sh 'cargo build'
                            } else {
                                ssh_sh "cargo dinghy --platform ${target.triple} build"
                            }
                       }

                       section(target, "test-${target.id}") {
                            if (targetName != 'ios') {
                                ssh_sh 'cargo test'
                            } else if (targetName != 'ios'){
                                ssh_sh "cargo dinghy --platform ${target.triple} test"
                            }
                       }
                   }
               } // end parallelize
               }

           } //end targetName
   } // end for targetsName

   parallel( jobs )

   if (build.isRelease) {
        machine(targets, "release") {
            section(targets, "Release") {
                performReleaseIfNeeded(build)
            }
        }
    }
}
