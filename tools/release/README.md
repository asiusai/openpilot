# openpilot releases

## Asius v0 build artifact

Run the manual `Asius v0 prebuilt` GitHub Actions workflow from `master`.
It uses a VamOS ARM64 runner with the `asius-v0-prebuilt` label and a working Adreno GPU.
Use a dedicated bench runner. The job builds in a temporary directory and preserves `/data/openpilot`.
It includes Chestnut models, compiled local GPU models, flattened submodules, and one Git commit.
The artifact uses the normal development Panda firmware. It does not install software.
After the build, a GitHub-hosted job publishes the snapshot to `asiusai/openpilot:master-ci`.
That branch contains Asius v0 binaries. Keep source development on `master`.
Each successful publish replaces the previous generated snapshot. The push checks that the branch has not changed concurrently.
Actions also keeps the archive, checksum, and build report for 14 days.
The build report records source, OS, dependency lock, model hashes, and sizes.
Native compilation and model checks do not replace full device and Chestnut validation.

```
## release checklist

### Go to staging
- [ ] make a GitHub issue to track release with this checklist
- [ ] create release master branch
  - [ ] create a branch from upstream master named `zerotentwo` for release `v0.10.2`
  - [ ] revert risky commits (double check with autonomy team)
  - [ ] push the new branch
- [ ] push to staging:
  - [ ] make sure you are on the newly created release master branch (`zerotentwo`)
  - [ ] run `BRANCH=devel-staging tools/release/build_stripped.sh`. Jenkins will then automatically build staging on device, run `test_onroad` and update the staging branch
- [ ] bump version on master: `openpilot/common/version.h` and `RELEASES.md`
- [ ] post on Discord, tag `@release crew`

### Go to release
- [ ] before going to release, test the following:
  - [ ] update from previous release -> new release
  - [ ] update from new release -> previous release
  - [ ] fresh install with `openpilot-test.comma.ai`
  - [ ] drive on fresh install
  - [ ] no submodules; only the Chestnut big driving model uses LFS
  - [ ] check MTBF, etc.
  - [ ] stress test passes in production
- [ ] publish the blog post
- [ ] `git reset --hard origin/release-mici-staging`
- [ ] tag the release: `git tag v0.X.X <commit-hash> && git push origin v0.X.X`
- [ ] create GitHub release
- [ ] final test install on `openpilot.comma.ai`
- [ ] update factory provisioning
- [ ] close out milestone and issue
- [ ] post on Discord, X, etc.
```
