open System
open System.Diagnostics

let exec (cmd: string) =
    let cmd = cmd.Replace("\"", "\"\"")
    Process.Start("/bin/bash", "-c \"" + cmd + "\"").WaitForExit()

[<EntryPoint>]
let main argv =
    exec """git add -A"""
    exec """git add -A"""
    exec """git commit -m 'update repo'"""
    exec """git push"""

    exec """mdbook build"""
    exec """rm -rf release/*""" // this won't delete the .git directory
    exec """cp -rp book/* release/"""

    exec """cd release; git add -A"""
    exec """cd release; git commit -m 'update book'"""
    exec """cd release; git push origin gh-pages"""
    
    0 // return an integer exit code