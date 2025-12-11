"""
Daemon script to run the bot continuously
Restarts automatically on crash
"""
import subprocess
import sys
import time
import os
from datetime import datetime

def run_bot():
    """Run the bot and restart on crash"""
    max_restarts = 1000  # Maximum restart attempts
    restart_count = 0
    restart_delay = 10  # Wait 10 seconds before restart
    
    print("=" * 60)
    print("🤖 AI Pump/Dump Detection Bot - Daemon Mode")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Bot will restart automatically on crash")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    print()
    
    while restart_count < max_restarts:
        try:
            print(f"\n🔄 Starting bot (attempt {restart_count + 1})...")
            print("-" * 60)
            
            # Run the bot using subprocess (more reliable)
            try:
                process = subprocess.Popen(
                    [sys.executable, "main.py"],
                    stdout=None,  # Direct output
                    stderr=subprocess.STDOUT
                )
                
                # Wait for process to finish
                return_code = process.wait()
            except Exception as e:
                print(f"Error starting bot: {e}")
                return_code = 1
            
            if return_code == 0:
                print("\n✅ Bot stopped normally")
                break
            else:
                restart_count += 1
                print(f"\n⚠️  Bot crashed (exit code: {return_code})")
                print(f"🔄 Restarting in {restart_delay} seconds... (restart {restart_count}/{max_restarts})")
                time.sleep(restart_delay)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Daemon stopped by user")
            if 'process' in locals():
                process.terminate()
            break
        except Exception as e:
            restart_count += 1
            print(f"\n❌ Error: {e}")
            print(f"🔄 Restarting in {restart_delay} seconds... (restart {restart_count}/{max_restarts})")
            time.sleep(restart_delay)
    
    if restart_count >= max_restarts:
        print(f"\n❌ Maximum restart attempts ({max_restarts}) reached. Stopping.")

if __name__ == "__main__":
    run_bot()

