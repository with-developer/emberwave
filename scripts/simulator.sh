# Sourced by the scripts that run things in an iOS Simulator
# (check_xcframework.sh, check_ios_app.sh); not a script of its own.
#
# use_simulator sets `device` to the UDID of a booted iPhone simulator, or
# boots an iPhone of the newest iOS runtime and shuts it down when the
# script exits. Needs Xcode with an iOS Simulator runtime, and jq.

use_simulator() {
    device=$(xcrun simctl list devices booted --json |
        jq -r '[.devices[][] | select(.name | startswith("iPhone"))][0].udid // empty')
    if [ -z "$device" ]; then
        device=$(xcrun simctl list devices available --json | jq -r '
            [.devices | to_entries[] | select(.key | contains("SimRuntime.iOS-"))
             | (.key | sub(".*iOS-"; "") | split("-") | map(tonumber)) as $version
             | .value[] | select(.name | startswith("iPhone"))
             | {udid, $version}]
            | sort_by(.version) | last | .udid // empty')
        if [ -z "$device" ]; then
            echo "no iPhone simulator: install an iOS runtime in Xcode" >&2
            exit 1
        fi
        trap 'xcrun simctl shutdown "$device" || true' EXIT
        xcrun simctl bootstatus "$device" -b
    fi
    echo "== in the simulator $(xcrun simctl list devices --json |
        jq -r --arg udid "$device" '.devices | to_entries[] | .key as $runtime
            | .value[] | select(.udid == $udid) | "\(.name), \($runtime | sub(".*SimRuntime."; ""))"')"
}
