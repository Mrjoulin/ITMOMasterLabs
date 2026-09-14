import React, { useState, useRef, useEffect } from 'react';
import { User, Area } from '../types';

interface SidebarProps {
  currentPage: string;
  onNavigate: (page: string) => void;
  user: User | null;
  onLogout: () => void;
  areas: Area[];
  onCreateArea: (name: string, color: string) => Promise<void>;
  activeTasksCount: number;
}

const Sidebar: React.FC<SidebarProps> = ({ currentPage, onNavigate, user, onLogout, areas, onCreateArea, activeTasksCount }) => {
  const [showLogout, setShowLogout] = useState(false);
  const [isCreatingArea, setIsCreatingArea] = useState(false);
  const [newAreaName, setNewAreaName] = useState('');
  
  const colors = [
    'bg-red-500', 'bg-orange-500', 'bg-amber-500',
    'bg-green-500', 'bg-teal-500', 'bg-blue-500',
    'bg-indigo-500', 'bg-purple-500', 'bg-pink-500'
  ];

  const getNextAvailableColor = () => {
    const usedColors = areas.map(area => area.color);
    return colors.find(color => !usedColors.includes(color)) || colors[0];
  }

  const [newAreaColor, setNewAreaColor] = useState(getNextAvailableColor());
  const [showColorPalette, setShowColorPalette] = useState(false);
  
  const profileRef = useRef<HTMLDivElement>(null);
  const creationRef = useRef<HTMLDivElement>(null);

  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: 'dashboard' },
    { id: 'tasks', label: 'Tasks', icon: 'check_circle', badge: activeTasksCount },
    { id: 'notes', label: 'Notes', icon: 'description' },
  ];

  useEffect(() => {
    setNewAreaColor(getNextAvailableColor());
  }, [areas]);

  // Close popup listeners
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
        // Logout popup
        if (profileRef.current && !profileRef.current.contains(event.target as Node)) {
            setShowLogout(false);
        }

        // Create Area cancel on click outside
        if (isCreatingArea && creationRef.current && !creationRef.current.contains(event.target as Node)) {
            // Also check if clicking color palette options (which might be in portal or just absolute)
            // In this structure, palette is child of creationRef, so it should be fine.
            resetCreation();
        }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isCreatingArea]);

  const resetCreation = () => {
      setIsCreatingArea(false);
      setNewAreaName('');
      setNewAreaColor(getNextAvailableColor());
      setShowColorPalette(false);
  };

  const handleCreateArea = async () => {
      if (newAreaName.trim()) {
          await onCreateArea(newAreaName, newAreaColor);
          resetCreation();
      }
  };

  return (
    <aside className="w-64 bg-white dark:bg-[#1a202c] border-r border-slate-200 dark:border-slate-800 flex flex-col h-screen fixed left-0 top-0 z-10 hidden md:flex">
      {/* Logo */}
      <div className="h-16 flex items-center px-6 border-b border-slate-100 dark:border-slate-800/50">
        <img src="/logo.png" alt="FocusFlow logo" className="w-8 h-8 rounded-lg mr-3 object-contain" />
        <h1 className="text-lg font-bold tracking-tight text-slate-900 dark:text-white">FocusFlow</h1>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto py-6 px-3 space-y-1">
        {navItems.map((item) => {
            const isActive = currentPage === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onNavigate(item.id)}
                className={`w-full flex items-center px-3 py-2.5 text-sm font-medium rounded-lg transition-colors group ${
                  isActive 
                    ? 'text-primary bg-primary/10' 
                    : 'text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800'
                }`}
              >
                <span className={`material-icons text-[20px] mr-3 ${isActive ? 'text-primary' : 'text-slate-400 group-hover:text-primary'}`}>
                  {item.icon}
                </span>
                {item.label}
                {(item.badge > 0) && (
                  <span className={`ml-auto py-0.5 px-2 rounded-full text-xs font-semibold ${isActive ? 'bg-primary text-white' : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400'}`}>
                    {item.badge}
                  </span>
                )}
              </button>
            );
        })}

        <div className="pt-4 mt-4 border-t border-slate-100 dark:border-slate-800 px-3">
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Areas</p>
            {areas.map((area) => {
                const isActive = currentPage === `area:${area.id}`;
                return (
                    <button 
                        key={area.id} 
                        onClick={() => onNavigate(`area:${area.id}`)}
                        className={`w-full flex items-center px-2 py-2 text-sm rounded-lg group transition-colors ${
                            isActive 
                            ? 'bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-white' 
                            : 'text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800'
                        }`}
                    >
                        <span className={`w-2.5 h-2.5 rounded-full ${area.color} mr-3`}></span>
                        {area.name}
                    </button>
                )
            })}
            
            {/* Create Area UI */}
            <div className="mt-2" ref={creationRef}>
                {!isCreatingArea ? (
                     <button 
                        onClick={() => setIsCreatingArea(true)}
                        className="w-full flex items-center px-2 py-2 text-sm text-slate-400 hover:text-primary rounded-lg hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors group"
                    >
                        <span className="material-icons">add</span>
                        Create area
                    </button>
                ) : (
                    <div className="flex items-center px-1 py-1 bg-slate-50 dark:bg-slate-800/50 rounded-lg border border-slate-200 dark:border-slate-700 animate-fade-in-up">
                        {/* Buttons on left
                        <button onClick={resetCreation} className="p-0 text-slate-400 hover:text-red-500 rounded-full hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
                            <span className="material-icons">close</span>
                        </button>
                        <button onClick={handleCreateArea} className="p-0 text-slate-400 hover:text-green-500 rounded-full hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors mr-1">
                            <span className="material-icons">check</span>
                        </button>
                        */}
                        {/* Color Dot with Palette */}
                        <div className="relative">
                            <div
                                data-testid="color-dot"
                                onClick={() => setShowColorPalette(!showColorPalette)}
                                className={`w-4 h-4 rounded-full ${newAreaColor} cursor-pointer hover:scale-110 transition-transform flex-shrink-0 mr-1 ring-1 ring-offset-1 ring-slate-300 dark:ring-slate-600 ring-offset-transparent`}
                            ></div>

                            {showColorPalette && (
                                <div data-testid="color-palette" className="absolute bottom-full left-0 right-auto mb-2 p-2 bg-white dark:bg-slate-800 rounded-lg shadow-xl border border-slate-200 dark:border-slate-700 z-50 grid grid-cols-3 gap-2 w-24">
                                    {colors.map(color => (
                                        <div
                                            key={color}
                                            onClick={() => { setNewAreaColor(color); setShowColorPalette(false); }}
                                            className={`w-5 h-5 rounded-full ${color} cursor-pointer hover:scale-110 transition-transform border border-slate-200 dark:border-slate-600`}
                                        ></div>
                                   ))}
                                </div>
                            )}
                        </div>
                        <input 
                            autoFocus
                            value={newAreaName}
                            onChange={(e) => setNewAreaName(e.target.value)}
                            onKeyDown={(e) => {
                                if (e.key === 'Enter') handleCreateArea();
                                if (e.key === 'Escape') resetCreation();
                            }}
                            className="w-full bg-transparent border-none text-sm px-2 py-1 text-slate-900 dark:text-white placeholder-slate-400 focus:ring-0 min-w-0"
                            placeholder="Name"
                        />
                    </div>
                )}
            </div>
        </div>
      </nav>

      {/* User Profile */}
      <div className="p-4 border-t border-slate-200 dark:border-slate-800 relative" ref={profileRef}>
        {showLogout && (
             <div className="absolute bottom-full left-4 right-4 mb-2 bg-white dark:bg-slate-800 rounded-lg shadow-xl border border-slate-100 dark:border-slate-700 py-1 z-20 animate-fade-in-up">
                 <button 
                    onClick={onLogout}
                    className="w-full text-left px-4 py-2 text-sm text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 flex items-center gap-2"
                 >
                     <span className="material-icons">logout</span>
                     Log out
                 </button>
             </div>
        )}
        
        <div 
            onClick={() => setShowLogout(!showLogout)}
            className="flex items-center w-full p-2 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors cursor-pointer"
        >
          {user ? (
             <>
                <div className="h-9 w-9 rounded-full bg-primary flex items-center justify-center text-white font-bold">
                    {user.full_name.charAt(0).toUpperCase()}
                </div>
                <div className="ml-3 overflow-hidden text-left">
                    <p className="text-sm font-medium text-slate-900 dark:text-white truncate">{user.full_name}</p>
                    <p className="text-xs text-slate-500 dark:text-slate-400 truncate">{user.email}</p>
                </div>
             </>
          ) : (
            <div className="h-9 w-9 bg-slate-200 rounded-full animate-pulse"></div>
          )}
          <span className="material-icons text-slate-400 ml-auto text-sm">more_vert</span>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;