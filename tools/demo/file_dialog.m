#import <Foundation/Foundation.h>
#import <AppKit/NSOpenPanel.h>
#import <AppKit/NSApplication.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#include <string.h>

const char *open_file_dialog(const char *title, const char *default_dir) {
    @autoreleasepool {
        NSOpenPanel *panel = [NSOpenPanel openPanel];
        [panel setCanChooseFiles:YES];
        [panel setCanChooseDirectories:NO];
        [panel setAllowsMultipleSelection:NO];

        if (title) {
            [panel setMessage:[NSString stringWithUTF8String:title]];
        }

        if (default_dir) {
            NSString *dir = [NSString stringWithUTF8String:default_dir];
            if (![dir isAbsolutePath]) {
                NSString *cwd = [[NSFileManager defaultManager] currentDirectoryPath];
                dir = [cwd stringByAppendingPathComponent:dir];
            }
            [panel setDirectoryURL:[NSURL fileURLWithPath:dir]];
        }

        // macOS 11+ UTType API
        if (@available(macOS 11.0, *)) {
            UTType *ggufType = [UTType typeWithFilenameExtension:@"gguf"];
            if (ggufType) {
                [panel setAllowedContentTypes:@[ggufType]];
            }
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            [panel setAllowedFileTypes:@[@"gguf"]];
#pragma clang diagnostic pop
        }

        NSModalResponse response = [panel runModal];
        if (response == NSModalResponseOK) {
            NSURL *url = [[panel URLs] firstObject];
            if (url) {
                return strdup([[url path] UTF8String]);
            }
        }
        return NULL;
    }
}
