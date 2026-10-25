# One shared server for several MCP clients

By default each MCP client starts its own FusionVisionMCP over stdio. That needs no setup, and it is what
the `.mcpb` bundle and the README's client snippets do. The cost shows up when several clients run at once:
Claude Code with a few chats open, Cline, Claude Desktop, another IDE. Each spawns a separate server, each
loads its own copy of the models it uses (the default captioner alone peaks near 8.7 GB), and each one's
memory manager can only see the others as "less free memory".

`--transport streamable-http` runs **one** server that every client connects to by URL. One copy of each model
serves all of them, and one memory manager sees every call.

## Run it

```shell
fusion-vision-mcp --transport streamable-http            # http://127.0.0.1:8765/mcp
fusion-vision-mcp --transport streamable-http --port 9000 --memory-mode aggressive
```

Every other option (`--memory-mode`, `--memory-budget`, `--device`, the model choices) works as it does
over stdio. Clients no longer start the server, so something has to keep it running: see below.

## Point the clients at it

| Client | Entry |
| --- | --- |
| Claude Code (`~/.claude.json` or `claude mcp add`) | `"fusionvision": {"type": "http", "url": "http://127.0.0.1:8765/mcp"}`, or `claude mcp add --transport http fusionvision http://127.0.0.1:8765/mcp` |
| Cline (`cline_mcp_settings.json`) | `"fusionvision": {"type": "streamableHttp", "url": "http://127.0.0.1:8765/mcp"}` |
| Claude Desktop, other clients | whatever form the client uses for a remote/HTTP MCP server, with the URL above |

A client that connects while the server is down reports it as failing to connect; start the server and
reconnect.

## Keep it running

**Windows.** Task Scheduler starts a program at logon but does not restart one that exits, so
`scripts/run_shared_server.ps1` does that (with a backoff, logging to
`%LOCALAPPDATA%\FusionVisionMCP\shared_server.log`). Register it to start at logon with no console window:

```powershell
$script = (Resolve-Path scripts\run_shared_server.ps1).Path
$action = New-ScheduledTaskAction -Execute "conhost.exe" `
    -Argument "--headless powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$script`""
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit 0 -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName "FusionVisionMCP shared server" -Action $action -Trigger $trigger -Settings $settings
Start-ScheduledTask -TaskName "FusionVisionMCP shared server"
```

Remove it with `Unregister-ScheduledTask -TaskName "FusionVisionMCP shared server"`.

**macOS** (launchd restarts it itself): `~/Library/LaunchAgents/com.fusionvisionmcp.shared.plist`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.fusionvisionmcp.shared</string>
  <key>ProgramArguments</key><array>
    <string>/Users/YOU/.local/bin/fusion-vision-mcp</string>
    <string>--transport</string><string>streamable-http</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardErrorPath</key><string>/tmp/fusion-vision-mcp.log</string>
</dict></plist>
```

then `launchctl load ~/Library/LaunchAgents/com.fusionvisionmcp.shared.plist`.

**Linux** (systemd user unit): `~/.config/systemd/user/fusion-vision-mcp.service`

```ini
[Unit]
Description=FusionVisionMCP shared server

[Service]
ExecStart=%h/.local/bin/fusion-vision-mcp --transport streamable-http
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
```

then `systemctl --user enable --now fusion-vision-mcp`.

## Security

The tools read image files by path, so anything that can send this server a request can have it read and
describe any image the user running it can read. Over stdio only the client that started it can talk to
it; over HTTP, anything that reaches the port can. So:

- **It listens on 127.0.0.1 by default**, unreachable from other machines. A `--host` that is not loopback
  logs a warning at startup; don't, unless the network in between is yours alone.
- **Web pages can't reach it through the browser.** On loopback the MCP SDK checks the `Host` and `Origin`
  headers, so a page that tricks the browser into calling `127.0.0.1` under another name (DNS rebinding) is
  refused.
- **There is no authentication.** Other programs, and on a shared machine other users' programs, can reach
  a loopback port. If the machine has accounts you don't trust, keep stdio.

## Trade-offs

One server is one point of failure: if it stops, every client loses the tools until it is restarted (the
Windows script and launchd/systemd restart it within seconds). And calls from different clients now share
one machine's worth of memory deliberately: a heavy call from one client can wait for another's to finish,
which `server_status` shows.
